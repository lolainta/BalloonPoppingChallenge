"""PPO with the attention policy (RL2). Optional behaviour-cloning warm start from dev/bc_collect.py demos.

.venv/bin/python dev/train_ppo2.py --envs 90 --timesteps 4000000 --out runs/ppo5 --bc 'data/bc/*.npz'
Env vars for the reach-model loop: T1DATA=<dir> (log decisions), SLMODEL=<npz> (hot-reloaded).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse, csv, glob, shutil, sys, time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dev.rl2_env import GuidanceEnv  # noqa: E402
from dev.rl2_net import AttnPolicy, export  # noqa: E402


def bc_pretrain(policy, patterns, epochs, log):
    d = [np.load(f) for f in sorted({f for p in patterns for f in glob.glob(p)})]  # shell may pre-expand the glob
    X = torch.tensor(np.concatenate([x["X"] for x in d]))
    A = torch.tensor(np.concatenate([x["A"] for x in d])).long()
    S = torch.tensor(np.concatenate([x["S"] for x in d])).long()
    R = torch.tensor(np.concatenate([x["R"] for x in d])).float()
    dev = next(policy.net.parameters()).device
    X, A, S, R = X.to(dev), A.to(dev), S.to(dev), R.to(dev)
    n = len(X); idx = torch.randperm(n, device=dev); val, tr = idx[: n // 20], idx[n // 20 :]
    opt = torch.optim.Adam(policy.net.parameters(), lr=1e-3)
    for ep in range(epochs):
        policy.net.train()
        for j in tr[torch.randperm(len(tr), device=dev)].split(512):
            tl, sl, v = policy.net(X[j])
            loss = (torch.nn.functional.cross_entropy(tl, A[j]) + 0.5 * torch.nn.functional.cross_entropy(sl, S[j])
                    + 0.05 * torch.nn.functional.mse_loss(v, R[j]))
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            tl, sl, v = policy.net(X[val])
            msg = (f"bc epoch {ep + 1}: target acc {(tl.argmax(1) == A[val]).float().mean():.3f}  "
                   f"stretch acc {(sl.argmax(1) == S[val]).float().mean():.3f}  value mae {(v - R[val]).abs().mean():.2f}  (n={n})")
        print(msg, flush=True); log.write(msg + "\n"); log.flush()


class BestEpisodes:
    """Log every training episode with at least MIN_POPS pops (seed, last pop time, step) to best_episodes.csv.
    Reference only: actions are sampled and weights change mid-episode, so these runs are not reproducible."""

    MIN_POPS = int(os.environ.get("BEST_MIN_POPS", 9))  # training-episode log threshold

    def __init__(self, out, export_fn, name):
        self.out, self.export_fn, self.name, self.best, self.best_lp = Path(out), export_fn, name, -1.0, 1e9
        f = self.out / "best_episodes.csv"
        if f.exists():
            for line in f.read_text().splitlines()[1:]:
                self.best = max(self.best, float(line.split(",")[1]))
        else:
            f.write_text("step,pops,last_pop,seed,snapshot\n")
        self.f = open(f, "a")

    def check(self, step, infos, policy_net):
        for info in infos:
            ep = info.get("episode")
            if not ep or ep["r"] < self.MIN_POPS:
                continue
            snap = ""  # log only: sampled training episodes cannot be reproduced, so no weights are kept for them
            self.f.write(f"{step},{ep['r']:.0f},{info.get('last_pop')},{info.get('seed')},{snap}\n"); self.f.flush()


class Checkpoint(BaseCallback):
    def __init__(self, out, every, name="rl2_policy"):
        super().__init__()
        self.out, self.every, self.n, self.t0 = Path(out), every, 0, time.time()
        self.best = BestEpisodes(out, lambda n, p: export(n, p), name)
        self.w = csv.writer(f := open(self.out / "progress.csv", "a", newline="")); self.f = f

    def _on_step(self):
        self.best.check(self.num_timesteps, self.locals.get("infos", []), self.model.policy.net)
        return True

    def _on_rollout_end(self):
        self.n += 1
        buf = self.model.ep_info_buffer
        rew = np.mean([e["r"] for e in buf]) if buf else float("nan")
        self.w.writerow([self.num_timesteps, len(buf), f"{rew:.3f}", f"{time.time() - self.t0:.0f}"]); self.f.flush()
        if time.time() - getattr(self, "t_saved", self.t0) >= 600:  # time-based: rollout size varies with the env count
            self.t_saved = time.time()
            self.model.save(self.out / "last.zip")
            export(self.model.policy.net, self.out / f"{self.best.name}.npz")
            export(self.model.policy.net, self.out / f"{self.best.name}_{self.num_timesteps}.npz")
            sl = os.environ.get("SLMODEL")
            if sl and os.path.exists(sl):  # the policy sees SL outputs: pin the version it was trained against
                shutil.copy2(sl, self.out / f"rl2_sl_{self.num_timesteps}.npz")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--timesteps", type=int, default=100_000)
    ap.add_argument("--n_steps", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--ent", type=float, default=0.003)
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--bc", nargs="+", help="BC demo npz files or globs")
    ap.add_argument("--bc_epochs", type=int, default=10)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume")
    ap.add_argument("--device", default="cpu", help="learner device: cpu or cuda (envs always run on CPU)")
    ap.add_argument("--launch", action="store_true", help="RL3: the policy also decides when to launch")
    ap.add_argument("--init_from", help="RL2 model zip to copy weights from (RL3 warm start)")
    ap.add_argument("--remote", help="worker dir (dev/remote_envs.py serve ...): add those nodes' envs")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    if a.launch:  # RL3 swaps env, policy and export; the rest of the loop is shared
        from dev.rl3_env import GuidanceEnv3 as GuidanceEnv
        from dev.rl3_net import AttnPolicy3 as AttnPolicy, export, init_from_rl2
        globals().update(export=export)
    venv = SubprocVecEnv([lambda: GuidanceEnv() for _ in range(a.envs)], start_method="fork")
    if a.remote:
        from dev.remote_envs import MultiNodeVecEnv
        venv = MultiNodeVecEnv(venv, a.remote)
        print(f"total envs: {venv.num_envs} ({a.envs} local)", flush=True)
    venv = VecMonitor(venv)
    batch = venv.num_envs * a.n_steps // 4
    if a.resume:
        model = PPO.load(a.resume, env=venv, device=a.device)
        model.batch_size = batch  # env count may differ from the saved run
    else:
        model = PPO(AttnPolicy, venv, n_steps=a.n_steps, batch_size=batch, n_epochs=4,
                    learning_rate=a.lr, ent_coef=a.ent, gamma=0.99, gae_lambda=0.95, clip_range=0.1,
                    seed=0, device=a.device, verbose=1)
        if a.init_from:
            src = PPO.load(a.init_from, device="cpu").policy.net.state_dict()
            init_from_rl2(model.policy.net, src) if a.launch else model.policy.net.load_state_dict(src)
            print(f"initialised from {a.init_from}", flush=True)
        if a.bc:
            with open(out / "bc.log", "a") as log:
                bc_pretrain(model.policy, a.bc, a.bc_epochs, log)
            export(model.policy.net, out / "rl2_policy_bc.npz")
    name = "rl3_policy" if a.launch else "rl2_policy"
    model.learn(a.timesteps, callback=Checkpoint(out, a.every, name), reset_num_timesteps=not a.resume)
    model.save(out / "last.zip"); export(model.policy.net, out / f"{name}.npz")

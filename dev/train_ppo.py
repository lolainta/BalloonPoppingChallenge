"""PPO for RL guidance (dev/rl_env.py). Exports the policy MLP to numpy for agents/rl_agent.py.

.venv/bin/python dev/train_ppo.py --envs 90 --timesteps 3000000 --out runs/ppo1 [--resume runs/ppo1/last.zip]
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse, csv, sys, time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dev.rl_env import GuidanceEnv  # noqa: E402


def export(model, path):
    """Policy MLP (tanh hidden layers + action logits) -> npz in the layout RLAgent loads."""
    layers = [m for m in model.policy.mlp_extractor.policy_net if isinstance(m, torch.nn.Linear)]
    layers.append(model.policy.action_net)
    out = {"n_layers": len(layers)}
    for i, m in enumerate(layers):
        out[f"W{i}"] = m.weight.detach().cpu().numpy().T
        out[f"b{i}"] = m.bias.detach().cpu().numpy()
    np.savez(path, **out)


class BestEpisodes:
    """Log every training episode with at least MIN_POPS pops (seed, last pop time, step) to best_episodes.csv,
    and snapshot the policy whenever the single-episode record improves (stochastic policy: the snapshot
    must be re-checked deterministically on that seed before relying on it)."""

    MIN_POPS = 8

    def __init__(self, out, export_fn, name):
        self.out, self.export_fn, self.name, self.best = Path(out), export_fn, name, -1.0
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
            snap = ""
            if ep["r"] > self.best:
                self.best = ep["r"]
                snap = f"{self.name}_best_{int(ep['r'])}_{info.get('seed')}.npz"
                self.export_fn(policy_net, self.out / snap)
            self.f.write(f"{step},{ep['r']:.0f},{info.get('last_pop')},{info.get('seed')},{snap}\n"); self.f.flush()


class Checkpoint(BaseCallback):
    def __init__(self, out, every):
        super().__init__()
        self.out, self.every, self.n, self.t0 = Path(out), every, 0, time.time()
        self.best = BestEpisodes(out, export, "rl_policy")
        self.log = open(self.out / "progress.csv", "a", newline="")
        self.w = csv.writer(self.log)

    def _on_step(self):
        self.best.check(self.num_timesteps, self.locals.get("infos", []), self.model)
        return True

    def _on_rollout_end(self):
        self.n += 1
        buf = self.model.ep_info_buffer
        rew = np.mean([e["r"] for e in buf]) if buf else float("nan")
        self.w.writerow([self.num_timesteps, len(buf), f"{rew:.3f}", f"{time.time() - self.t0:.0f}"]); self.log.flush()
        if self.n % self.every == 0:
            self.model.save(self.out / "last.zip")
            export(self.model, self.out / "rl_policy.npz")
            export(self.model, self.out / f"rl_policy_{self.num_timesteps}.npz")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--timesteps", type=int, default=100_000)
    ap.add_argument("--n_steps", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ent", type=float, default=0.01)
    ap.add_argument("--every", type=int, default=5, help="checkpoint every N rollouts")
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume")
    ap.add_argument("--warm", type=float, default=2.0, help="initial logit bias toward (nearest target, stretch 1.3); 0 = off")
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    venv = VecMonitor(SubprocVecEnv([lambda: GuidanceEnv() for _ in range(a.envs)], start_method="fork"))
    if a.resume:
        model = PPO.load(a.resume, env=venv, device="cpu")
    else:
        model = PPO("MlpPolicy", venv, n_steps=a.n_steps, batch_size=a.envs * a.n_steps // 2, n_epochs=5,
                    learning_rate=a.lr, ent_coef=a.ent, gamma=0.99, gae_lambda=0.95, seed=0, device="cpu", verbose=1,
                    policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128]), activation_fn=torch.nn.Tanh))
    if a.warm and not a.resume:  # warm start without demos: bias the policy toward the GNC-like default action
        from BalloonPoppingGymEnv.agents.rl_agent import K, STRETCH
        with torch.no_grad():
            b = model.policy.action_net.bias
            b.zero_(); b[0] = a.warm; b[K + STRETCH.index(1.3)] = a.warm
    model.learn(a.timesteps, callback=Checkpoint(a.out, a.every), reset_num_timesteps=not a.resume)
    model.save(Path(a.out) / "last.zip"); export(model, Path(a.out) / "rl_policy.npz")

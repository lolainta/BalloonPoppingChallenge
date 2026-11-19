"""D: worker-side log-prob (async_ppo.act) vs the learner's recomputation (same formulas as async_ppo.learn) on one
sampled scenario-4 episode, pad and post-launch rows. RL3_SCEN=4 RL3_ATTC=1 RL3_RES=0.25 RL3_T_MIN=1"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys, warnings
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dev import async_ppo as ap  # noqa: E402
from dev.rl3_env import GuidanceEnv3  # noqa: E402
from dev.rl3_net import Net3, load_npz  # noqa: E402
from BalloonPoppingGymEnv.agents.rl3_agent import RL3Agent  # noqa: E402

T = Path(__file__).parent / "tmp"
warnings.filterwarnings("ignore")
SRC, OUT = (("rl3_policy_res5.npz", "rl3_policy_resD5.npz") if ap.NR == 5 else ("rl3_policy_res0.npz", "rl3_policy_resD.npz"))
P = dict(np.load(T / SRC)); rng0 = np.random.default_rng(0)
P["res.w"] = (0.05 * rng0.standard_normal(P["res.w"].shape)).astype(np.float32); P["res.b"] = np.array([0.2, -0.1, 0.1, 0.3, -0.2][:ap.NR], np.float32)
np.savez(T / OUT, **P)

env = GuidanceEnv3(seed_lo=100_000, seed_hi=10_000_000)
env.agent_kw.update({k[4:]: float(v) for k, v in P.items() if k.startswith("gnc.")}); env.agent_kw.update({f"res_{k}": float(P[f"res.{k}"]) for k in ("bound", "sbound", "nbound") if f"res.{k}" in P})
agent = RL3Agent.__new__(RL3Agent); agent.net = P
rng = np.random.default_rng(1)
x, _ = env.reset(seed=int(sys.argv[1]) if len(sys.argv) > 1 else 12345)
X, A, LP, done = [], [], [], False
while not done:
    a, lp, v = ap.act(agent, x, True, rng)
    X.append(x); A.append(a); LP.append(lp)
    x, r, done, _, _ = env.step(a)
X, A, LP = np.array(X, np.float32), np.array(A), np.array(LP)
print(f"rows {len(X)} (pad {int((X[:, ap.LAUNCHED_COL] < 0.5).sum())}), action width {A.shape[1]}, pops {env.info.get('popped_count')}")

for dev, dt in [("cpu", torch.float32), ("cpu", torch.float64)] + ([("cuda", torch.float32)] if torch.cuda.is_available() else []):
    net = Net3(); load_npz(net, T / OUT); net.to(dev, dt)
    from dev.rl3_net import RES_LS
    O, ACT = torch.tensor(X, device=dev, dtype=dt), torch.tensor(A, dtype=dt, device=dev)
    with torch.no_grad():
        logits, _ = net(O)
        lp, k = 0.0, 0
        for i, m in enumerate(ap.dims(True)):
            z = torch.log_softmax(logits[:, k: k + m], -1); k += m
            lp = lp + z.gather(1, ACT[:, i: i + 1].long())[:, 0]
        ls = net.attc_logstd; pre = (O[:, ap.LAUNCHED_COL] < 0.5).float()  # = learn.gauss
        lp = lp + (-0.5 * ((ACT[:, 3:5] - logits[:, k: k + 2]) / ls.exp()) ** 2 - ls - 0.5 * ap.LOG2PI).sum(-1) * pre
        rlp = ap.res_gauss(net, logits[:, -ap.NR:], ACT[:, -ap.NR:], O)[0]  # = learn.rgauss
        lp = (lp + rlp).cpu().numpy()
    d = np.abs(lp - LP); padm = X[:, ap.LAUNCHED_COL] < 0.5
    rw = np.zeros(len(X))  # the worker's residual part, recomputed exactly as act() does
    lsw = np.clip(P["res.logstd"], *RES_LS)
    for i in np.flatnonzero(~padm):
        agent.forward(X[i]); u = A[i, -ap.NR:].astype(np.float32)
        rw[i] = float((-0.5 * ((u - agent.res_u) / np.exp(lsw)) ** 2 - lsw - 0.5 * ap.LOG2PI).sum())
    print(f"{dev} {dt}: post rows: residual part max diff {np.abs(rw - rlp.cpu().numpy())[~padm].max():.2e}, "
          f"categorical part max diff {np.abs((LP - rw) - (lp - rlp.cpu().numpy()))[~padm].max():.2e}")
    print(f"{dev} {dt}: max |worker logp - learner logp|  pad {d[padm].max():.2e}  post-launch {d[~padm].max():.2e}  "
          f"(residual part on post rows: mean {rlp.cpu().numpy()[~padm].mean():.3f}, pad residual columns all 0: {not A[padm, -ap.NR:].any()})")

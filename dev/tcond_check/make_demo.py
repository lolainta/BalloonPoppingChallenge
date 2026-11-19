"""D helper: fly one stochastic scenario-4 training episode (dev/rl3_env.py + async_ppo.act) with POLICY.npz and save it as a
teacher demo (obs, act, rew; = dev/search_teacher.py's format) into RUN/demos_in, so a short learner run exercises SIL
(its own >= SIL_MIN episodes take longer than a mini run on this machine).   python dev/tcond_check/make_demo.py POLICY.npz RUN"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys, warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dev import async_ppo as ap  # noqa: E402
from dev.rl3_env import GuidanceEnv3  # noqa: E402
from BalloonPoppingGymEnv.agents.rl3_agent import RL3Agent  # noqa: E402

assert Path(ap.__file__).resolve().is_relative_to(ROOT), ap.__file__
warnings.filterwarnings("ignore")
P = dict(np.load(sys.argv[1]))
env = GuidanceEnv3(seed_lo=100_000, seed_hi=10_000_000)
env.agent_kw.update({k[4:]: float(v) for k, v in P.items() if k.startswith("gnc.")})
env.agent_kw.update({f"res_{k}": float(P[f"res.{k}"]) for k in ("bound", "sbound", "nbound") if f"res.{k}" in P})
agent = RL3Agent.__new__(RL3Agent); agent.net = P
rng = np.random.default_rng(3)
x, _ = env.reset(seed=777)
O, A, R, done = [], [], [], False
while not done:
    a, _, _ = ap.act(agent, x, True, rng)
    O.append(x); A.append(a)
    x, r, done, _, _ = env.step(a); R.append(r)
out = Path(sys.argv[2]) / "demos_in"; out.mkdir(parents=True, exist_ok=True)
np.savez(out / "tc_demo.npz", obs=np.array(O, np.float32), act=np.array(A), rew=np.array(R, np.float32))
print(f"demo: {len(O)} rows, action width {np.shape(A)[1]}, pops {sum(R):.0f}, post-launch targets {sorted(set(int(a[0]) for a, o in zip(A, O) if o[ap.LAUNCHED_COL] >= 0.5))[:12]}")

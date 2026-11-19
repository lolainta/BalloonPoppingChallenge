"""B (data): K training episodes (async_ppo.act, the worker's sampling) of POLICY on one scenario-4 seed, the world
reset once (RL3_REUSE) -> OUT.npz with the worker rows (obs, act, logp, val, rew, done).
source dev/hoent_check/env.sh; $PY dev/hoent_check/rollout.py POLICY.npz SEED K OUT.npz"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys, time, warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
pol, seed, K, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
os.environ.update(SEED_LIST=str(seed), RL3_REUSE=str(K), PREFETCH="0")
from dev import async_ppo as ap  # noqa: E402
from dev.rl3_env import GuidanceEnv3  # noqa: E402
from BalloonPoppingGymEnv.agents import rl3_agent  # noqa: E402
from BalloonPoppingGymEnv.agents.rl3_agent import RL3Agent  # noqa: E402

for m in (ap, rl3_agent):
    assert Path(m.__file__).resolve().is_relative_to(ROOT), m.__file__
warnings.filterwarnings("ignore")
P = dict(np.load(pol))
env = GuidanceEnv3(seed_lo=100_000, seed_hi=10_000_000)
env.agent_kw.update({k[4:]: float(v) for k, v in P.items() if k.startswith("gnc.")})
env.agent_kw.update({f"res_{k}": float(P[f"res.{k}"]) for k in ("bound", "sbound", "nbound") if f"res.{k}" in P})
agent = RL3Agent.__new__(RL3Agent); agent.net = P
rng = np.random.default_rng(seed)
rows = {k: [] for k in ("obs", "act", "logp", "val", "rew", "done")}
t0 = time.time()
for e in range(K):
    x, info = env.reset(seed=seed if e == 0 else None)
    done, pops = False, 0.0
    while not done:
        a, lp, v = ap.act(agent, x, True, rng)
        x2, r, done, _, inf = env.step(a)
        for k, val in zip(rows, (x, a, lp, v, r, done)):
            rows[k].append(val)
        pops += r; x = x2
    print(f"seed {info['seed']} episode {e} pops {pops:.0f} rows so far {len(rows['obs'])} ({time.time() - t0:.0f} s)", flush=True)
np.savez(out, **{k: np.array(v) for k, v in rows.items()})

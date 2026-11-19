"""D3: env episode built with bound 0 (and the episode prefetched before the reload), then a worker-style reload to a policy
with the res head (bound 0.25) mid-episode: per post-launch step, sampled u / its log-prob / the applied residual agree.
RL3_SCEN=4 RL3_ATTC=1 RL3_RES=0.25 RL3_T_MIN=1"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys, warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dev import async_ppo as ap  # noqa: E402
from dev.rl3_env import GuidanceEnv3  # noqa: E402
from BalloonPoppingGymEnv.agents.rl3_agent import RL3Agent, RES_LS  # noqa: E402

warnings.filterwarnings("ignore")
T = Path(__file__).parent / "tmp"
B, P = dict(np.load(T / "rl3_policy_86630400.npz")), dict(np.load(T / "rl3_policy_resD.npz"))
env = GuidanceEnv3()
env.agent_kw.update({k[4:]: float(v) for k, v in B.items() if k.startswith("gnc.")})  # worker reload of the base policy
agent = RL3Agent.__new__(RL3Agent); agent.net = B
rng = np.random.default_rng(3)
x, _ = env.reset(seed=4242)  # also spawns the next (prefetched) episode, still with bound 0
built = [env.ag.res_bound]
rows, post_n, reloaded, ep = [], 0, False, 0
while ep < 2:
    if post_n == 10 and not reloaded:  # = worker reload(): new weights + agent_kw["res_bound"]
        agent.net = P; env.agent_kw["res_bound"] = float(P["res.bound"]); reloaded = True
    post = x[ap.LAUNCHED_COL] >= 0.5
    a, lp, v = ap.act(agent, x, True, rng)
    cap = {}
    s0 = env.ag.set_res
    env.ag.set_res = lambda u, s0=s0: (cap.update(u=np.asarray(u, float).copy(), at=env.ag.a_thrust), s0(u))[1]
    lpr = 0.0
    if "res.w" in agent.net and post:
        ls = np.clip(agent.net["res.logstd"], *RES_LS); u = a[-3:].astype(np.float32)
        lpr = float((-0.5 * ((u - agent.res_u) / np.exp(ls)) ** 2 - ls - 0.5 * ap.LOG2PI).sum())
    ag = env.ag
    x, r, done, _, _ = env.step(a)
    del ag.set_res
    if post:
        post_n += 1
        app = ag.res.copy()
        exp = 0.25 * cap["at"] * np.tanh(a[-3:]) if len(a) == 8 else np.zeros(3)
        rows.append((ep, reloaded, len(a), lpr, ag.res_bound, bool(app.any()), float(np.abs(app - exp).max()), bool(np.abs(a[-3:] if len(a) == 8 else 0).sum() > 0)))
    if done:
        ep += 1
        if ep < 2:
            x, _ = env.reset(); built.append(env.ag.res_bound)
            print(f"episode 2 = the prefetched one, built with bound {built[-1]} (kwargs at its spawn)")
print("bound the agents were built with:", built)
for e in (0, 1):
    for rl in (False, True):
        R = [r for r in rows if r[0] == e and r[1] == rl]
        if R:
            print(f"episode {e + 1} {'after' if rl else 'before'} the reload: {len(R)} post-launch steps; action widths {sorted({r[2] for r in R})}; "
                  f"u sampled (non-zero) on {sum(r[7] for r in R)}; residual log-prob non-zero on {sum(r[3] != 0 for r in R)}; "
                  f"env bound {sorted({r[4] for r in R})}; residual applied on {sum(r[5] for r in R)}; "
                  f"max |applied - 0.25*a_thrust*tanh(u)| {max(r[6] for r in R):.1e}")

"""Diagnosis: every engagement (a target held until popped / switched) -> closest true distance, popped or not.
AGENT=rl3 .venv/bin/python dev/engage_diag.py --scenario 4 --seeds 10000-10023 --kw policy_path=... lag_comp=1"""
import os, sys, contextlib, io, warnings, logging, argparse, multiprocessing as mp
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dev.run_seeds import parse_kw, set_param
from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters
from BalloonPoppingGymEnv.agents.rl3_agent import RL3Agent


def run(a):
    seed, sc, kw = a
    warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
    kw = dict(kw); params, given = load_scenario_parameters(sc)
    for k in [k for k in kw if k.startswith("P.")]:
        set_param(params, k[2:], kw.pop(k))
    env = BalloonPoppingEnv(render_mode=None, parameters=params)
    with contextlib.redirect_stdout(io.StringIO()):
        obs, _ = env.reset(seed=seed)
    ag = RL3Agent(given, **kw)
    eng, cur, done = [], None, False  # cur = [target, t0, dmin, t_dmin]
    while not done:
        act = ag.get_action(obs)
        obs, r, term, trunc, info = env.step(act); done = term or trunc
        t = float(obs["simulation_time"]); tg = ag.target if ag.launched() and ag._burning(t) else None
        st = obs["balloon_status"][:, 0]; rs = info["rocket_states"]
        if cur is not None and (tg != cur[0] or st[cur[0]] != 1 or done):
            eng.append((cur[0], cur[1], t, cur[2], st[cur[0]] == 2))
            cur = None
        if tg is not None and st[tg] == 1 and not np.isnan(rs[0]):
            d = float(np.linalg.norm(obs["balloon_states"][tg, :3] - np.asarray(rs[:3], float)))
            if cur is None:
                cur = [tg, t, d, t]
            elif d < cur[2]:
                cur[2], cur[3] = d, t
    return seed, info["popped_count"], eng


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--scenario", type=int, default=4); ap.add_argument("--seeds", default="10000-10023")
    ap.add_argument("--kw", nargs="*", default=[]); a = ap.parse_args()
    lo, hi = map(int, a.seeds.split("-")); kw = parse_kw(a.kw)
    with mp.Pool(min(len(os.sched_getaffinity(0)), hi - lo + 1)) as pool:
        res = pool.map(run, [(s, a.scenario, kw) for s in range(lo, hi + 1)])
    pops = [p for _, p, _ in res]; E = [e for _, _, es in res for e in es]
    hit = [e for e in E if e[4]]; miss = [e for e in E if not e[4]]
    near = [e for e in miss if e[3] < 6.0]
    print(f"scenario {a.scenario} {a.kw}: mean pops {np.mean(pops):.2f}  engagements {len(E) / len(res):.1f}/ep  "
          f"popped {len(hit) / len(res):.2f}/ep  abandoned {len(miss) / len(res):.1f}/ep  near-miss(<6 m) {len(near) / len(res):.2f}/ep")
    if hit:
        print(f"  popped: mean engagement {np.mean([e[2] - e[1] for e in hit]):.2f}s")
    if near:
        print("  near-miss closest distances:", np.round(np.percentile([e[3] for e in near], [10, 50, 90]), 2))

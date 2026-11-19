"""How often does the agent fly past the balloon it is chasing, and why (DECISION_LOG §188)?
Per target attempt (same agent.cmd target, while burning): closest true distance, popped or not, and at the closest
point the navigation error |p_hat - p_true|, the agent's own estimated miss distance, and the speeds.
AGENT kwargs as dev/run_seeds.py --kw.  Writes one CSV row per attempt.
  .venv/bin/python dev/miss_probe.py --seeds 80400-80447 --csv logs/miss_probe.csv --kw policy_path=... sl_path=...
"""
import argparse, contextlib, csv, io, logging, multiprocessing as mp, os, sys, warnings
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dev.run_seeds import parse_kw

R_POP = 1.5  # (m) balloon radius = pop distance of the swept rocket path


def run(a):
    seed, kw = a
    warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
    from BalloonPoppingGymEnv.agents.rl2_sl import load_sl
    from BalloonPoppingGymEnv.agents.rl3_agent import RL3Agent
    from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv
    from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters
    params, given = load_scenario_parameters(int(os.environ.get("SC", 4)))  # (SC=40: the finale scenario)
    env = BalloonPoppingEnv(render_mode=None, parameters=params)
    with contextlib.redirect_stdout(io.StringIO()):
        obs, info = env.reset(seed=seed)
    agent = RL3Agent(given, **kw)
    if kw.get("sl_path"):
        agent.sl = load_sl(kw["sl_path"])
    # the agent hands over to its next target ~0.5 s before contact (the intercept finishes on momentum), so an
    # attempt stays open for HOLD s after the target changes: popped = that balloon popped while the attempt was open
    HOLD = 1.0
    rows, open_, done = [], {}, False
    while not done:
        obs, r, te, tr, info = env.step(agent.get_action(obs))
        done = te or tr
        t = float(obs["simulation_time"])
        s = np.asarray(info["rocket_states"], float)
        st = obs["balloon_status"][:, 0]
        tgt = agent.cmd[0] if (agent.cmd is not None and agent.p_hat is not None and not np.isnan(s[0])) else None
        if tgt is not None and st[tgt] == 1 and tgt not in open_:
            open_[tgt] = dict(seed=seed, i=int(tgt), t0=t, dmin=np.inf, end=None)
        for i, a in list(open_.items()):
            if i != tgt and a["end"] is None:
                a["end"] = t
            if st[i] == 2 or done or (a["end"] is not None and t > a["end"] + HOLD):
                a["popped"] = int(st[i] == 2)
                rows.append(open_.pop(i))
                continue
            b = np.asarray(obs["balloon_states"][i], float)
            d = float(np.linalg.norm(s[:3] - b[:3]))
            if d < a["dmin"]:
                a.update(dmin=d, t_min=t, nav=float(np.linalg.norm(agent.p_hat - s[:3])),
                         est=float(np.linalg.norm(agent.p_hat - b[:3])), spd=float(np.linalg.norm(s[3:6])),
                         relv=float(np.linalg.norm(s[3:6] - b[3:6])), burning=int(agent._burning(t)))
    return seed, info["popped_count"], rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="80400-80447"); ap.add_argument("--csv", required=True)
    ap.add_argument("--kw", nargs="*", default=[]); ap.add_argument("--procs", type=int, default=0)
    a = ap.parse_args()
    lo, _, hi = a.seeds.partition("-"); seeds = range(int(lo), int(hi or lo) + 1)
    kw = parse_kw(a.kw)
    from dev.seed_registry import record
    record(seeds, "miss_probe", " ".join(a.kw), 4)
    with mp.Pool(a.procs or len(os.sched_getaffinity(0))) as pool, open(a.csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["seed", "target", "t0", "t_min", "dmin", "popped", "nav", "est", "spd", "relv", "burning"])
        tot = []
        for seed, n, rows in pool.imap_unordered(run, [(s, kw) for s in seeds]):
            tot.append(n)
            for r in rows:
                if np.isfinite(r["dmin"]):
                    w.writerow([seed, r["i"], round(r["t0"], 2), round(r.get("t_min", 0), 2), round(r["dmin"], 3), r["popped"],
                                round(r.get("nav", 0), 3), round(r.get("est", 0), 3), round(r.get("spd", 0), 2),
                                round(r.get("relv", 0), 2), r.get("burning", 0)])
            fh.flush()
    print(f"episodes {len(tot)}  mean pops {np.mean(tot):.3f}")

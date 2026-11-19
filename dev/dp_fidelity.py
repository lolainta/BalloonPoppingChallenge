"""DP planner fidelity / timing (dev/plans/dp_planner.md M1): planned vs flown legs, per-replan time, shadow slope.

One env reset per seed, every config flies a deep copy (as dev/sweep_gnc.py). Configs should set dp_log=1.
  dp=1      : committed first legs (consecutive replans with the same target; the plan at the first one) flown to
              completion (popped = hit, passed = miss; switched away = abandoned, not counted): hit rate, planned vs
              actual arrival time and velocity (true rocket velocity at the pop)
  dp=2      : override rate;  dp_shadow=1: slope of realised pops over the plan's span against its value J
  all       : per-replan ms p50 / p95, guard fires, pops
.venv/bin/python dev/dp_fidelity.py --seeds 14900-14915 --procs 8 --base policy_path=runs/s4cont/rl3_policy_8332800.npz \
    sl_path=runs/s4cont/rl2_sl_8332800.npz lag_comp=1 margin=0 bwind=2 --cfg "dp=1 dp_log=1"
"""
import os, sys, copy, contextlib, io, warnings, logging, argparse, multiprocessing as mp
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dev.run_seeds import parse_kw
from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters


def run(a):
    seed, sc, base, cfgs = a
    warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
    from BalloonPoppingGymEnv.agents.rl3_agent import RL3Agent
    params, given = load_scenario_parameters(sc)
    env0 = BalloonPoppingEnv(render_mode=None, parameters=params)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            obs0, _ = env0.reset(seed=seed)
    except Exception:
        return seed, None
    out = []
    for c in cfgs:
        env, obs = copy.deepcopy(env0), copy.deepcopy(obs0)
        ag = RL3Agent(given, **{**base, **c})
        pops, st0, done = [], obs["balloon_status"][:, 0].copy(), False
        while not done:
            obs, r, te, tr, info = env.step(ag.get_action(obs)); done = te or tr
            st = obs["balloon_status"][:, 0]
            for i in np.flatnonzero((st == 2) & (st0 != 2)):  # (balloon, time, true rocket velocity)
                pops.append((int(i), float(obs["simulation_time"]), np.asarray(info["rocket_states"][3:6], float)))
            st0 = st.copy()
        burn_end = (ag.launched_at or 0.0) + ag.burn_time
        out.append(dict(pops=pops, rec=ag.dp_rec, ms=list(ag.dp_ms), fired=ag.dp_fired, n=info["popped_count"], burn_end=burn_end))
    return seed, out


def legs(r):
    """Committed first legs of one dp=1 episode -> list of (outcome, dt_first, dt_final, dv_first, dv_final, tau_first,
    planned - actual speed at the pop (first plan), lead of the final commitment, first plan kept to the pop). first =
    the plan when the target was chosen; final = the plan that set the arrival time finally flown (t_arr unchanged)."""
    rec, pt = r["rec"], {i: (t, v) for i, t, v in r["pops"]}
    out, i = [], 0
    while i < len(rec):
        g = rec[i]["tgt"]
        j = i
        while j + 1 < len(rec) and rec[j + 1]["tgt"] == g:
            j += 1
        grp = [x for x in rec[i:j + 1] if x["V"] is not None]
        nxt = rec[j + 1] if j + 1 < len(rec) else None
        i = j + 1
        if g is None or not grp:
            continue
        f, n = grp[0], len(grp) - 1
        while n > 0 and grp[n - 1]["t_arr"] == grp[-1]["t_arr"]:
            n -= 1
        l = grp[n]
        t_end = nxt["t"] if nxt else np.inf
        if g in pt and f["t"] <= pt[g][0] <= t_end + 1e-9:
            tp, vp = pt[g]
            out.append(("hit", f["t_arr"] - tp, l["t_arr"] - tp, np.linalg.norm(f["V"] - vp), np.linalg.norm(l["V"] - vp),
                        f["t_arr"] - f["t"], np.linalg.norm(f["V"]) - np.linalg.norm(vp), l["t_arr"] - l["t"], float(n == 0)))
        elif (nxt and nxt["passed"]) or (nxt is None and g not in pt):
            out.append(("miss",) + (np.nan,) * 4 + (f["t_arr"] - f["t"], np.nan, np.nan, np.nan))
        else:
            out.append(("abandon",) + (np.nan,) * 4 + (f["t_arr"] - f["t"], np.nan, np.nan, np.nan))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", type=int, default=4); ap.add_argument("--seeds", default="14900-14915")
    ap.add_argument("--base", nargs="*", default=[]); ap.add_argument("--cfg", nargs="+", default=["dp=1 dp_log=1"])
    ap.add_argument("--procs", type=int, default=8); a = ap.parse_args()
    lo, hi = map(int, a.seeds.split("-")); base = parse_kw(a.base)
    cfgs = [parse_kw(c.split()) for c in a.cfg]
    with mp.Pool(a.procs) as pool:
        res = [r for r in pool.imap_unordered(run, [(s, a.scenario, base, cfgs) for s in range(lo, hi + 1)]) if r[1]]
    print(f"scenario {a.scenario} base {a.base} seeds {len(res)}")
    for j, (c, kw) in enumerate(zip(a.cfg, cfgs)):
        R = [r[1][j] for r in res]
        ms = np.concatenate([r["ms"] for r in R]) if any(r["ms"] for r in R) else np.zeros(1)
        n_rep = sum(len(r["ms"]) for r in R)
        print(f"[{c}] pops {np.mean([r['n'] for r in R]):.3f}  replans {n_rep}  ms p50 {np.median(ms):.1f} p95 {np.percentile(ms, 95):.1f}"
              f" max {ms.max():.0f}  guard fires {sum(r['fired'] for r in R) / max(n_rep, 1):.2%}")
        if int(kw.get("dp", 0)) == 1:
            L = [x for r in R for x in legs(r)]
            oc = np.array([x[0] for x in L])
            H = np.array([x[1:] for x in L if x[0] == "hit"], float).reshape(-1, 8)
            nh, nm = (oc == "hit").sum(), (oc == "miss").sum()
            print(f"  legs {len(L)}: hit {nh} miss {nm} abandoned {(oc == 'abandon').sum()}  hit rate {nh / max(nh + nm, 1):.3f}"
                  f"  median planned leg {np.median([x[5] for x in L]):.2f} s")
            if len(H):
                print(f"  hits: |t_arr - actual| median first {np.median(np.abs(H[:, 0])):.3f} s (signed {np.median(H[:, 0]):+.3f})"
                      f"  final {np.median(np.abs(H[:, 1])):.3f} s;  |dv| median first {np.median(H[:, 2]):.2f}  final {np.median(H[:, 3]):.2f} m/s"
                      f"  (planned - actual speed {np.median(H[:, 5]):+.2f}; final commitment {np.median(H[:, 6]):.2f} s before arrival)")
                k = H[H[:, 7] > 0]  # the first plan's arrival time was flown unchanged
                if len(k):
                    print(f"  hits flown on the first plan's t_arr ({len(k)}): |dt| median {np.median(np.abs(k[:, 0])):.3f} s"
                          f" (signed {np.median(k[:, 0]):+.3f})  |dv| median {np.median(k[:, 2]):.2f} m/s  planned leg {np.median(k[:, 4]):.2f} s")
                for lo_, hi_ in ((0, 2), (2, 4), (4, 99)):  # by planned leg time
                    h = H[(H[:, 4] >= lo_) & (H[:, 4] < hi_)]
                    if len(h):
                        print(f"    legs {lo_}-{hi_} s: n {len(h)}  dt first median {np.median(h[:, 0]):+.3f}  |dt| {np.median(np.abs(h[:, 0])):.3f}"
                              f"  |dv| {np.median(h[:, 2]):.2f}  speed err {np.median(h[:, 5]):+.2f}")
        if int(kw.get("dp", 0)) == 2:
            rec = [x for r in R for x in r["rec"]]
            nov = sum(x["tgt"] is not None and x["tgt"] != x["pi"] for x in rec)
            print(f"  policy decisions {len(rec)}  overridden {nov} ({nov / max(len(rec), 1):.1%})")
        if int(kw.get("dp_shadow", 0)):
            X = [(x["J"], x["Ep"], sum(x["t"] < t <= x["t_end"] for _, t, _ in r["pops"])) for r in R for x in r["rec"]
                 if x["J"] is not None and x["t_end"] is not None]
            X = np.array(X, float)
            if len(X) > 2:  # realised pops over the best plan's span vs its value J and its expected pops Ep
                for c_, nm in ((0, "J"), (1, "Ep")):
                    print(f"  shadow: realised pops over the plan span vs {nm}: slope {np.polyfit(X[:, c_], X[:, 2], 1)[0]:.3f}"
                          f"  corr {np.corrcoef(X[:, c_], X[:, 2])[0, 1]:.3f}  (n {len(X)}, mean {nm} {X[:, c_].mean():.2f}, realised {X[:, 2].mean():.2f})")

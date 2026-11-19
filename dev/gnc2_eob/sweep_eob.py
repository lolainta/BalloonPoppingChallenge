"""Paired end-of-burn (eob) sweep, dev/sweep_gnc.py plus per-seed flight details: one env reset per seed, every config
flies a deep copy of the same world. Per config: pops, pops after burnout ("coast"), pops in the last 2 s of the burn,
whether the eob mode engaged, and the closest approach (m) after burnout to the last target before burnout (0: popped after burnout; any config).
AGENT=rl3 .venv/bin/python dev/gnc2_eob/sweep_eob.py --scenario 4 --seeds 30000-30191 \
    --base policy_path=runs/s4bw4/rl3_policy_25478400.npz sl_path=runs/s4bw4/rl2_sl_25478400.npz \
    --cfg "" "eob=1" --out dev/gnc2_eob/res_s4.json
"""
import os, sys, copy, contextlib, io, json, warnings, logging, argparse, multiprocessing as mp
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dev.run_seeds import parse_kw, set_param
from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters


def fly(env, obs, ag):
    pops = coast = late = 0  # pops counted from popped_count
    eob_tgt, eob_t, dmin, t_bo, alive_bo = None, None, np.inf, None, None
    done = False
    while not done:
        obs, r, te, tr, info = env.step(ag.get_action(obs)); done = te or tr
        t = float(obs["simulation_time"])
        if ag.launched_at is not None and t_bo is None:
            t_bo = ag.launched_at + ag.burn_time - ag.dt
        if ag.target is not None and t_bo is not None and t < t_bo:  # the last target before burnout
            eob_tgt = int(ag.target)
            if getattr(ag, "eob_f", None) is not None and eob_t is None:
                eob_t = t  # the eob mode first engaged
        n = int(info["popped_count"]) - pops; pops += n
        if n and t_bo is not None and t >= t_bo:
            coast += n
        elif n and t_bo is not None and t >= t_bo - 2.0:
            late += n
        if eob_tgt is not None and t < t_bo:
            alive_bo = obs["balloon_status"][eob_tgt, 0] == 1
        if eob_tgt is not None and t_bo <= t <= t_bo + 3.0 and alive_bo is not False:
            b = obs["balloon_states"][eob_tgt]; rs = info["rocket_states"]
            if obs["balloon_status"][eob_tgt, 0] == 2:
                if alive_bo:  # popped after burnout
                    dmin = 0.0
                alive_bo = False
            elif np.isfinite(rs[:3]).all():
                dmin = min(dmin, float(np.linalg.norm(np.asarray(b[:3], float) - np.asarray(rs[:3], float))))
    return dict(pops=int(info["popped_count"]), coast=coast, late=late, eob=eob_t is not None,
                eob_t=None if eob_t is None else round(eob_t - t_bo, 2), dmin=None if not np.isfinite(dmin) else round(dmin, 2))


def run(a):
    seed, sc, base, cfgs, pkw = a
    warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
    from BalloonPoppingGymEnv.agents.rl3_agent import RL3Agent
    params, given = load_scenario_parameters(sc)
    for k, v in pkw.items():
        set_param(params, k, v)
    env0 = BalloonPoppingEnv(render_mode=None, parameters=params)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            obs0, _ = env0.reset(seed=seed)
    except Exception:
        return seed, None
    out = []
    for c in cfgs:
        env, obs = copy.deepcopy(env0), copy.deepcopy(obs0)
        out.append(fly(env, obs, RL3Agent(given, **{**base, **c})))
    return seed, out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", type=int, default=4); ap.add_argument("--seeds", default="30000-30191")
    ap.add_argument("--base", nargs="*", default=[]); ap.add_argument("--cfg", nargs="+", default=[""])
    ap.add_argument("--procs", type=int, default=0); ap.add_argument("--out", default="")
    a = ap.parse_args()
    lo, hi = map(int, a.seeds.split("-")); base = parse_kw(a.base)
    pkw = {k[2:]: base.pop(k) for k in [k for k in base if k.startswith("P.")]}
    cfgs = [parse_kw(c.split()) for c in a.cfg]
    with mp.Pool(a.procs or len(os.sched_getaffinity(0))) as pool:
        res = sorted(r for r in pool.imap_unordered(run, [(s, a.scenario, base, cfgs, pkw) for s in range(lo, hi + 1)]) if r[1])
    if a.out:
        json.dump(dict(scenario=a.scenario, base=a.base, cfgs=a.cfg, res=res), open(a.out, "w"))
    P = np.array([[o["pops"] for o in r[1]] for r in res], float)
    C = np.array([[o["coast"] for o in r[1]] for r in res], float)
    L = np.array([[o["late"] for o in r[1]] for r in res], float)
    print(f"scenario {a.scenario} base {a.base} seeds {len(res)}")
    for j, c in enumerate(a.cfg):
        d, dc = P[:, j] - P[:, 0], C[:, j] - C[:, 0]
        eo = [r[1][j] for r in res if r[1][j]["eob"]]
        dm = [r[1][j]["dmin"] for r in res if r[1][j]["dmin"] is not None]
        print(f"  [{c or 'base'}]  mean {P[:, j].mean():.3f}  diff {d.mean():+.3f} +- {d.std() / np.sqrt(len(d)):.3f}"
              f"  (+{(d > 0).sum()}/-{(d < 0).sum()})  coast {C[:, j].mean():.3f} (diff {dc.mean():+.3f} +- {dc.std() / np.sqrt(len(d)):.3f})"
              f"  late2s {L[:, j].mean():.3f}  eob engaged {len(eo)}  end target hit after burnout {sum(x == 0 for x in dm)} / missed by <5 m {sum(0 < x < 5 for x in dm)} of {len(dm)}  median dmin {np.median(dm) if dm else float('nan'):.2f}")

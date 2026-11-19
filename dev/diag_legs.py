"""Dev: why do short engagements miss? Per engagement: how it ended, swept closest approach, and at that
moment the navigation error, attitude lag (commanded force vs true thrust axis) and limiter saturation.

AGENT=gnc|beam uv run python dev/diag_legs.py 1957083 0 205 14 [--kw k=v ...]
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse, contextlib, io, logging, multiprocessing as mp, sys, warnings
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from BalloonPoppingGymEnv.agents.beam_agent import BeamAgent
from BalloonPoppingGymEnv.agents.gnc_agent import GNCAgent, quat_to_R
from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters

R_BALLOON = 1.5


def seg_min(r0, r1):
    """Min |r| on the segment r0 -> r1 (env uses swept segments)."""
    d = r1 - r0
    u = np.clip(-(r0 @ d) / max(d @ d, 1e-12), 0, 1)
    return np.linalg.norm(r0 + u * d)


def run(args):
    seed, kw = args
    warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
    params, given = load_scenario_parameters(1)
    env = BalloonPoppingEnv(render_mode=None, parameters=params)
    with contextlib.redirect_stdout(io.StringIO()):
        obs, info = env.reset(seed=seed)
    ag = (BeamAgent if os.environ.get("AGENT", "beam") == "beam" else GNCAgent)(given, **kw)
    eng, cur, done = [], None, False
    while not done:
        act = ag.get_action(obs)
        tg = ag.target if ag.launched_at is not None and ag._burning(float(obs["simulation_time"])) else None
        b_prev = obs["balloon_states"].copy()
        obs, r, te, tr, info = env.step(act)
        done = te or tr
        t = float(obs["simulation_time"])
        x = info["rocket_states"]
        if np.isnan(x[0]):
            continue
        p, v = x[:3], x[3:6]
        if cur is not None and (tg != cur["b"] or done):
            st = obs["balloon_status"][cur["b"], 0]
            closing = (cur["r_last"] @ cur["vrel_last"]) < 0
            cur["end"] = "pop" if st == 2 else ("gone" if st != 1 else ("switch_closing" if closing else "miss_passed"))
            eng.append(cur); cur = None
        if tg is None:
            continue
        b = obs["balloon_states"][tg]
        rel = p - b[:3]
        if cur is None:
            cur = dict(b=tg, t0=t, d0=np.linalg.norm(rel), cpa=1e9, r_prev=rel)
        c = seg_min(cur["r_prev"], rel)
        f = ag.debug.get("f")
        if c < cur["cpa"]:
            z_true = quat_to_R(x[6:10])[:, 2]
            lag = np.degrees(np.arccos(np.clip(z_true @ (f / np.linalg.norm(f)), -1, 1))) if f is not None else np.nan
            nav = np.linalg.norm(ag.p_hat - p) if ag.p_hat is not None else np.nan
            sat = f is not None and np.linalg.norm(f) >= 0.995 * ag.a_thrust
            # miss split: along relative velocity (timing) vs perpendicular (lateral)
            vr = v - b[3:]
            along = abs(rel @ vr) / max(np.linalg.norm(vr), 1e-9)
            cur.update(cpa=c, t_cpa=t, lag=lag, nav=nav, sat=sat, vrel=np.linalg.norm(vr),
                       lateral=np.sqrt(max(rel @ rel - along**2, 0)), speed=np.linalg.norm(v))
        # attitude lag and saturation averaged over the whole engagement
        if f is not None:
            z_true = quat_to_R(x[6:10])[:, 2]
            cur.setdefault("lags", []).append(np.degrees(np.arccos(np.clip(z_true @ (f / np.linalg.norm(f)), -1, 1))))
            cur.setdefault("sats", []).append(np.linalg.norm(f) >= 0.995 * ag.a_thrust)
        cur["r_prev"], cur["r_last"], cur["vrel_last"] = rel, rel, v - b[3:]
    return seed, info["popped_count"], eng


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("seeds", nargs="+", type=int)
    ap.add_argument("--kw", nargs="*", default=[])
    a = ap.parse_args()
    kw = {k: float(v) for k, v in (it.split("=") for it in a.kw)}
    with mp.Pool(len(a.seeds)) as pool:
        res = pool.map(run, [(s, kw) for s in a.seeds])
    allE = []
    for seed, n, eng in res:
        print(f"seed {seed}: popped {n}, engagements {len(eng)}")
        allE += [e for e in eng if e["t_cpa" if "t_cpa" in e else "t0"] - e["t0"] >= 0.2]
    ends = {}
    for e in allE: ends[e["end"]] = ends.get(e["end"], 0) + 1
    print("engagement endings (>=0.2 s):", ends)
    short = [e for e in allE if e["d0"] <= 45 and e["end"] in ("pop", "miss_passed")]
    print(f"\nshort engagements (start <=45 m) that reached closest approach: {len(short)}")
    for kind in ("pop", "miss_passed"):
        E = [e for e in short if e["end"] == kind]
        if not E: continue
        g = lambda k: np.array([e.get(k, np.nan) for e in E], float)
        print(f"  {kind:12s} n={len(E):3d}  cpa med {np.nanmedian(g('cpa')):5.1f} m  lateral med {np.nanmedian(g('lateral')):5.1f} m"
              f"  vrel med {np.nanmedian(g('vrel')):4.0f} m/s  nav err med {np.nanmedian(g('nav')):4.2f} m"
              f"  att lag @cpa med {np.nanmedian(g('lag')):4.1f} deg  eng-mean lag {np.nanmedian([np.mean(e.get('lags',[np.nan])) for e in E]):4.1f} deg"
              f"  sat frac {np.nanmean([np.mean(e.get('sats',[0])) for e in E]):.2f}")
    miss = [e for e in allE if e["end"] == "miss_passed"]
    print("\nall passed-by misses: cpa distribution (m):", np.percentile([e["cpa"] for e in miss], [10, 25, 50, 75, 90]).round(1) if miss else "-")

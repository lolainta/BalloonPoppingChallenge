"""Post-burnout non-gravity acceleration (truth): along / across the velocity, vs the body-velocity angle, and the
component of the across part toward the body axis. .venv/bin/python dev/gnc2_eob/aero.py --seeds 30000-30003 --kw ..."""
import os, sys, contextlib, io, warnings, logging, argparse, multiprocessing as mp
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dev.run_seeds import parse_kw
from BalloonPoppingGymEnv.agents.gnc_agent import quat_to_R, G


def run(a):
    seed, sc, kw = a
    warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
    from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv
    from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters
    from BalloonPoppingGymEnv.agents.rl3_agent import RL3Agent
    params, given = load_scenario_parameters(sc)
    env = BalloonPoppingEnv(render_mode=None, parameters=params)
    with contextlib.redirect_stdout(io.StringIO()):
        obs, _ = env.reset(seed=seed)
    ag = RL3Agent(given, **kw)
    done, rows, prev = False, [], None
    while not done:
        obs, r, te, tr, info = env.step(ag.get_action(obs)); done = te or tr
        if ag.launched_at is None:
            continue
        t = float(obs["simulation_time"]); tb = ag.launched_at + ag.burn_time - ag.dt
        s = np.asarray(info["rocket_states"], float)
        if prev is not None and tb - 0.3 <= t <= tb + 3.0:
            a_ng = (s[3:6] - prev[3:6]) / ag.dt - G
            v = s[3:6]; u = v / np.linalg.norm(v); z = quat_to_R(s[6:10])[:, 2]
            along = a_ng @ u; perp = a_ng - along * u
            zp = z - (z @ u) * u
            ang = np.degrees(np.arccos(np.clip(z @ u, -1, 1)))
            tow = perp @ (zp / max(np.linalg.norm(zp), 1e-9))
            rows.append((t - tb, np.linalg.norm(v), ang, along, np.linalg.norm(perp), tow))
        prev = s
    return seed, np.array(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", type=int, default=4); ap.add_argument("--seeds", default="30000-30003")
    ap.add_argument("--kw", nargs="*", default=[]); a = ap.parse_args()
    lo, hi = map(int, a.seeds.split("-"))
    with mp.Pool(min(8, hi - lo + 1)) as pool:
        res = pool.map(run, [(s, a.scenario, parse_kw(a.kw)) for s in range(lo, hi + 1)])
    for seed, R in res:
        print(f"seed {seed}   t-tb   |v|   body-vel deg  along  |perp|  perp toward body axis")
        for row in R[::20]:
            print("   " + "  ".join(f"{x:7.2f}" for x in row))

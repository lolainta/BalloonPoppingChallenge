"""Balloon-trajectory dataset for prediction experiments (scenario 4, no rocket launched, so no pops).
Per seed: balloon_states (T, 100, 6) and balloon_status (T, 100) every 0.1 s for 60 s, plus the given parameters.
.venv/bin/python dev/traj/make_dataset.py 20000-20029 data/traj/s4   (one npz per seed)"""
import os, sys, contextlib, io, warnings, logging, multiprocessing as mp
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters

IDLE = {"launch": False, "launch_inclination_heading": np.array([90.0, 0.0]), "tvc": np.zeros(2), "roll": 0.0, "throttle": 1.0}


def run(a):
    seed, out = a
    warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
    params, _ = load_scenario_parameters(4)
    env = BalloonPoppingEnv(render_mode=None, parameters=params)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            obs, _ = env.reset(seed=seed)
    except Exception as e:
        return seed, f"reset failed: {e!r}"
    S, St, T = [], [], []
    for k in range(6000):
        if k % 10 == 0:
            S.append(obs["balloon_states"].astype(np.float32)); St.append(obs["balloon_status"][:, 0].astype(np.int8))
            T.append(float(obs["simulation_time"]))
        obs, *_ = env.step(IDLE)
    np.savez_compressed(os.path.join(out, f"seed{seed}.npz"), states=np.stack(S), status=np.stack(St), t=np.array(T))
    return seed, "ok"


if __name__ == "__main__":
    lo, hi = map(int, sys.argv[1].split("-")); out = sys.argv[2]; os.makedirs(out, exist_ok=True)
    with mp.Pool(min(len(os.sched_getaffinity(0)), hi - lo + 1)) as pool:
        for s, r in pool.imap_unordered(run, [(s, out) for s in range(lo, hi + 1)]):
            print(s, r, flush=True)

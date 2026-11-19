"""Flight-quality benchmark: re-fly the leaderboard replays' exact target order (dev/replay_teacher.py, strict) with
different GNC parameters; score = how many balloons we pop (8 replays, their total 104). One pass per config.

.venv/bin/python dev/gnc_sweep.py runs/ppo11/rl3_policy_<step>.npz [procs]
"""
import contextlib, io, os, sys, time
from glob import glob

import numpy as np

os.environ.setdefault("SLACK", "inf")
from dev import replay_teacher as rt
from dev import rl2_env

BASE = {}
CONFIGS = [("base", {})] + [(f"{k}={v}", {k: v}) for k, vs in [
    ("margin", [0.0, 0.25, 1.0]), ("nav_N", [1.5, 3.0, 4.0]), ("k_att", [3.0, 4.0]), ("max_rate", [2.5, 4.0]),
    ("k_rate", [15.0]), ("t_go_min", [0.1, 0.6])] for v in vs] + [
    ("excess along v 0.3", {"excess_dir": 2, "excess_frac": 0.3}), ("excess along v 0.6", {"excess_dir": 2, "excess_frac": 0.6}),
    ("excess up 0.5", {"excess_dir": 1, "excess_frac": 0.5}), ("speed 0.8 + nav_N 3", {"S": 1, "nav_N": 3.0})]


def one(ck, path, kw):
    kw = dict(kw); s = kw.pop("S", None)
    rt.FIX_S = None if s is None else str(s)
    info = rt.replay_info(path)
    params, given = rl2_env.load_scenario_parameters(1)
    env = rl2_env.BalloonPoppingEnv(render_mode=None, parameters=params)
    with contextlib.redirect_stdout(io.StringIO()):
        obs, _ = env.reset(seed=info["seed"])
    ag = rt.Teacher(given, info, policy_path=ck, sl_path=ck.replace("rl3_policy", "rl2_sl"), **kw)
    done = False
    while not done:
        obs, r, term, trunc, _ = env.step(ag.get_action(obs))
        done = term or trunc
    st = obs["balloon_status"][:, 0]
    return int((st == 2).sum())


if __name__ == "__main__":
    import multiprocessing as mp
    ck = sys.argv[1]; procs = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    files = sorted(glob("../lb_replays/[0-9]*_*.json"))
    tasks = [(ck, f, kw) for _, kw in CONFIGS for f in files]
    t0 = time.time()
    with mp.get_context("spawn").Pool(procs) as p:
        res = p.starmap(one, tasks)
    res = np.array(res).reshape(len(CONFIGS), len(files))
    print(f"{len(tasks)} games in {time.time() - t0:.0f}s; replays: " + " ".join(os.path.basename(f).split('.')[0] for f in files))
    for i in np.argsort(-res.sum(1), kind="stable"):
        print(f"{CONFIGS[i][0]:<22} total {res[i].sum():3d}/104  " + " ".join(f"{x:2d}" for x in res[i]))

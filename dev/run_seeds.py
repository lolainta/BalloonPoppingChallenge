"""Dev harness: run an agent on many seeds in parallel. Not used for evaluation.

uv run python dev/run_seeds.py --seeds 0-7 --kw launch_time=20 v_cruise=25
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")  # ponytail: one BLAS thread per worker, else 24 procs x 24 threads
import argparse
import logging
import multiprocessing as mp
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import os
from BalloonPoppingGymEnv.agents.beam_agent import BeamAgent
from BalloonPoppingGymEnv.agents.gnc_agent import GNCAgent, quat_to_R  # noqa: E402
from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv  # noqa: E402
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters  # noqa: E402


def clean_params(d):
    """Scenario without sensor noise / actuator lag (only its weather and balloons): isolates their effect."""
    for k, v in list(d.items()):
        if isinstance(v, dict):
            clean_params(v)
        elif k.endswith(("_noise_density", "_random_walk_density", "_accuracy")):
            d[k] = 0.0
        elif k.endswith("_constant_bias"):
            d[k] = [0.0] * len(v) if isinstance(v, (list, tuple)) else 0.0
        elif k.endswith("_time_constant"):
            d[k] = None
    return d


def set_param(d, key, val):
    """Override every scenario parameter named key (P.<key>=<val> on the command line; 'none' -> None, 'a,b,c' -> list,
    'true' / 'false' -> bool)."""
    if isinstance(val, str) and "," in val:
        val = [float(x) if "." in x else int(x) for x in val.split(",")]
    elif isinstance(val, str) and val in ("true", "false"):
        val = val == "true"
    if "." in key:  # a dotted path: 'rocket_body.mass' only under rocket_body (a plain 'mass' also hits the balloons')
        head, rest = key.split(".", 1)
        for k, v in d.items():
            if isinstance(v, dict):
                set_param(v, rest, val) if k == head else set_param(v, key, val)
        return
    for k, v in d.items():
        if k == key:
            d[k] = None if val == "none" else val
        elif isinstance(v, dict):
            set_param(v, key, val)


def run(args):
    seed, scenario, kw = args
    kw = dict(kw); scenario = int(kw.pop("scenario", scenario)); clean = kw.pop("clean", 0)  # evaluation options, not agent kwargs
    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)
    params, given = load_scenario_parameters(scenario)
    if clean:
        clean_params(params); clean_params(given)
    for k in [k for k in kw if k.startswith("P.")]:  # e.g. P.max_gust_speed=0: same random draws, no gust
        set_param(params, k[2:], kw.pop(k))
    rocket_nogust = kw.pop("rocket_nogust", 0)  # diagnosis: balloons fly in the gust, the rocket does not
    if rocket_nogust:
        from rocketpy import Environment
        add = Environment.add_wind_gust
        def rec(self, gx, gy):
            self._gust = (gx, gy); return add(self, gx, gy)
        Environment.add_wind_gust = rec
    env = BalloonPoppingEnv(render_mode=None, parameters=params)
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):
        obs, info = env.reset(seed=seed)
    if rocket_nogust and hasattr(env._rocketpy_env, "_gust"):  # balloons were simulated at reset on a copy
        gx, gy = env._rocketpy_env._gust
        add(env._rocketpy_env, gx * -1, gy * -1)
    from BalloonPoppingGymEnv.agents.rl_agent import RLAgent
    from BalloonPoppingGymEnv.agents.rl2_agent import RL2Agent
    from BalloonPoppingGymEnv.agents.rl3_agent import RL3Agent
    agent = {"beam": BeamAgent, "gnc": GNCAgent, "rl": RLAgent, "rl2": RL2Agent, "rl3": RL3Agent}[os.environ.get("AGENT", "beam")](given, **kw)
    if kw.get("sl_path"):
        from BalloonPoppingGymEnv.agents.rl2_sl import load_sl
        agent.sl = load_sl(kw["sl_path"])
    terminated = truncated = False
    t0, last_pop, att_err, coast = time.time(), None, 0.0, 0
    perr, verr = [], []  # navigation error vs truth while flying (p_hat / v_hat vs rocket state)
    while not (terminated or truncated):
        obs, r, terminated, truncated, info = env.step(agent.get_action(obs))
        if r > 0:
            last_pop = obs["simulation_time"]
            coast += not agent._burning(last_pop)  # popped after burnout
        s = info["rocket_states"]
        if not np.isnan(s[6]):  # debug only: nav attitude error vs truth
            z_est = quat_to_R(agent.q)[:, 2]
            z_true = quat_to_R(s[6:10])[:, 2]
            att_err = max(att_err, np.degrees(np.arccos(np.clip(z_est @ z_true, -1, 1))))
            if getattr(agent, "p_hat", None) is not None:
                perr.append(float(np.linalg.norm(agent.p_hat - np.asarray(s[:3], float))))
                verr.append(float(np.linalg.norm(agent.v_hat - np.asarray(s[3:6], float))))
    return seed, info["popped_count"], last_pop, att_err, time.time() - t0, obs["simulation_time"], env._episode_ending, agent.launched_at, coast, \
        (np.mean(perr) if perr else 0.0, np.max(perr) if perr else 0.0, np.mean(verr) if verr else 0.0)


def parse_kw(items):
    kw = {}
    for it in items:
        k, v = it.split("=")
        try:
            kw[k] = v if k.endswith("_path") else float(v)
        except ValueError:  # string options, e.g. launch_rule=median
            kw[k] = v
    return kw


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-7")
    ap.add_argument("--scenario", type=int, default=1)
    ap.add_argument("--kw", nargs="*", default=[])
    a = ap.parse_args()
    if "," in a.seeds or a.seeds.startswith("@"):  # explicit list: "3,17,42" or "@file" (one seed per line)
        txt = open(a.seeds[1:]).read() if a.seeds.startswith("@") else a.seeds
        seeds = [int(x) for x in txt.replace(",", " ").split()]
    else:
        lo, _, hi = a.seeds.partition("-")
        seeds = range(int(lo), int(hi or lo) + 1)
    kw = parse_kw(a.kw)
    from dev.seed_registry import record
    record(seeds, "run_seeds", f"AGENT={os.environ.get('AGENT', 'beam')} {' '.join(a.kw)}", int(kw.get("scenario", a.scenario)))
    with mp.Pool(min(len(seeds), len(os.sched_getaffinity(0)))) as pool:  # allocated cores (Slurm cpuset), not the whole node
        res = pool.map(run, [(s, a.scenario, kw) for s in seeds])
    for seed, n, lp, err, wall, t_end, ending, launch, coast, nav in res:
        print(f"seed {seed:3d}  popped {n:3d}  last_pop {lp}  max_att_err {err:.2f}deg  wall {wall:.0f}s  end {t_end:.1f}s {ending}  launch {launch}  coast {coast}"
              f"  nav_pos {nav[0]:.2f}/{nav[1]:.2f}m  nav_vel {nav[2]:.2f}m/s")
    print(f"{kw}  mean {np.mean([r[1] for r in res]):.2f}  min {min(r[1] for r in res)}")

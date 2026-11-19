"""Per-step action hash of deterministic scenario-4 RL3 runs (tc2 checks A/B/E), several policies in one pool.

.venv/bin/python dev/tc2_check/act_hash.py POLICY.npz@SL.npz@SEEDS [POLICY2@SL2@SEEDS2 ...] [--module single_file.py] [k=v ...]
Asserts that the agent modules come from the tree this script lives in (the venv's editable install points elsewhere).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import contextlib, hashlib, importlib.util, io, logging, multiprocessing as mp, sys, warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def run(args):
    seed, pol, sl, module, kw = args
    warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
    from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv
    from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters
    from BalloonPoppingGymEnv.agents import rl3_agent
    assert Path(rl3_agent.__file__).resolve().is_relative_to(ROOT), rl3_agent.__file__
    params, given = load_scenario_parameters(4)
    env = BalloonPoppingEnv(render_mode=None, parameters=params)
    with contextlib.redirect_stdout(io.StringIO()):
        obs, _ = env.reset(seed=seed)
    if module:  # single-file agent: weights + SL embedded
        spec = importlib.util.spec_from_file_location("sf_agent", module); m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        agent = m.RL3Agent(given, **kw)
    else:
        from BalloonPoppingGymEnv.agents.rl2_sl import load_sl
        agent = rl3_agent.RL3Agent(given, policy_path=pol, **kw)
        agent.sl = load_sl(sl)
    h, n, done = hashlib.sha256(), 0, False
    while not done:
        a = agent.get_action(obs)
        h.update(np.array([float(a["launch"]), *np.asarray(a["launch_inclination_heading"], float), *np.asarray(a["tvc"], float),
                           float(a["roll"]), float(a["throttle"])], np.float64).tobytes())
        obs, r, te, tr, info = env.step(a); n += 1; done = te or tr
    return Path(module or pol).name, seed, h.hexdigest()[:16], n, int(info["popped_count"])


if __name__ == "__main__":
    args, module, kw = sys.argv[1:], None, {}
    if "--module" in args:
        i = args.index("--module"); module = args[i + 1]; del args[i: i + 2]
    jobs = []
    for r in args:
        if "@" in r:
            pol, sl, seeds = r.split("@")
            jobs += [(int(s), pol, sl, module, kw) for s in seeds.split(",")]
        else:
            k, v = r.split("="); kw[k] = float(v)
    with mp.Pool(min(len(jobs), 6)) as p:
        for name, seed, hx, n, pops in p.map(run, jobs):
            print(f"{name}  seed {seed}  hash {hx}  steps {n}  popped {pops}  kw {kw}", flush=True)

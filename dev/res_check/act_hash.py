"""Per-step action hash of a deterministic RL3 run (res-head checks A/B/F).

.venv/bin/python dev/res_check/act_hash.py POLICY.npz SL.npz SEEDS [--module single_file.py] [k=v ...]
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import contextlib, hashlib, importlib.util, io, logging, multiprocessing as mp, sys, warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def run(args):
    seed, pol, sl, module, kw = args
    warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
    from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv
    from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters
    params, given = load_scenario_parameters(4)
    env = BalloonPoppingEnv(render_mode=None, parameters=params)
    with contextlib.redirect_stdout(io.StringIO()):
        obs, _ = env.reset(seed=seed)
    if module:  # single-file agent: weights + SL embedded
        spec = importlib.util.spec_from_file_location("sf_agent", module); m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        agent = m.RL3Agent(given, **kw)
    else:
        from BalloonPoppingGymEnv.agents.rl3_agent import RL3Agent
        from BalloonPoppingGymEnv.agents.rl2_sl import load_sl
        agent = RL3Agent(given, policy_path=pol, **kw)
        agent.sl = load_sl(sl)
    h, n, pops, done = hashlib.sha256(), 0, 0, False
    while not done:
        a = agent.get_action(obs)
        h.update(np.array([float(a["launch"]), *np.asarray(a["launch_inclination_heading"], float), *np.asarray(a["tvc"], float),
                           float(a["roll"]), float(a["throttle"])], np.float64).tobytes())
        obs, r, te, tr, info = env.step(a); n += 1; pops += r; done = te or tr
    return seed, h.hexdigest()[:16], n, int(info["popped_count"]), float(getattr(agent, "res_bound", 0.0))


if __name__ == "__main__":
    pol, sl, seeds, *rest = sys.argv[1:]
    module = None
    if rest and rest[0] == "--module":
        module, rest = rest[1], rest[2:]
    kw = {k: float(v) for k, v in (r.split("=") for r in rest)}
    seeds = [int(s) for s in seeds.split(",")]
    with mp.Pool(len(seeds)) as p:
        for seed, hx, n, pops, rb in p.map(run, [(s, pol, sl, module, kw) for s in seeds]):
            print(f"seed {seed}  hash {hx}  steps {n}  popped {pops}  res_bound {rb}", flush=True)

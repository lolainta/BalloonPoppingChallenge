"""Defect-1 mini runs: target-head entropy on fixed observations.
  collect OUT.npy POLICY SL SEED...: the features at every post-launch FULL decision of deterministic scenario-4 flights (evaluation agent)
  entropy OUT.npy A.npz B.npz ...: mean target entropy (nats) of each policy on the first 256 of them, all rows and the rows choosing a new
       target (no current target flagged: the RL3_NEWENT rows)
source dev/fast_check/env.sh; $PY dev/fast_check/ent_probe.py collect|entropy ..."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import contextlib, io, logging, sys, warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from BalloonPoppingGymEnv.agents import rl3_agent  # noqa: E402

assert Path(rl3_agent.__file__).resolve().is_relative_to(ROOT)
warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
mode, out = sys.argv[1:3]
if mode == "collect":
    from BalloonPoppingGymEnv.agents.rl2_sl import load_sl
    from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv
    from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters
    pol, sl = sys.argv[3:5]
    params, given = load_scenario_parameters(4)
    X = []
    for seed in map(int, sys.argv[5:]):
        env = BalloonPoppingEnv(render_mode=None, parameters=params)
        with contextlib.redirect_stdout(io.StringIO()):
            obs, _ = env.reset(seed=seed)
        ag = rl3_agent.RL3Agent(given, policy_path=pol); ag.sl = load_sl(sl)
        done = False
        while not done:
            t = float(obs["simulation_time"])
            if ag.p_hat is not None and ag._burning(t) and ag.needs_decision(obs):
                X.append(ag.features(obs))
            obs, r, te, tr, info = env.step(ag.get_action(obs)); done = te or tr
        print(f"seed {seed}: {len(X)} rows so far, pops {info['popped_count']}", flush=True)
    np.save(out, np.array(X, np.float32))
else:
    X = np.load(out)[:256]
    new = X[:, rl3_agent.G_DIM:].reshape(len(X), -1, 15)[:, :, 8].max(1) < 0.5
    for pol in sys.argv[3:]:
        ag = rl3_agent.RL3Agent.__new__(rl3_agent.RL3Agent); ag.net = dict(np.load(pol))
        H = []
        for x in X:
            tl = ag.forward(x)[0]; z = tl - tl.max(); p = np.exp(z) / np.exp(z).sum()
            H.append(-(p * np.log(np.maximum(p, 1e-30))).sum())
        H = np.array(H)
        print(f"{Path(pol).parent.name}/{Path(pol).name}: target entropy all {H.mean():.4f}  new-target rows ({int(new.sum())}) "
              f"{H[new].mean():.4f}", flush=True)

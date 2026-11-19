"""Dev: commanded vs achieved acceleration."""
import contextlib, io, logging, sys, warnings
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import os
from BalloonPoppingGymEnv.agents.beam_agent import BeamAgent
from BalloonPoppingGymEnv.agents.gnc_agent import GNCAgent, G
from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters
from dev.run_seeds import parse_kw
warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
kw = parse_kw(sys.argv[2:])
params, given = load_scenario_parameters(1)
env = BalloonPoppingEnv(render_mode=None, parameters=params)
with contextlib.redirect_stdout(io.StringIO()):
    obs, info = env.reset(seed=int(sys.argv[1]))
agent = (BeamAgent if os.environ.get('AGENT', 'beam') == 'beam' else GNCAgent)(given, **kw); terminated = truncated = False; vprev = None; rows = []
while not (terminated or truncated):
    act = agent.get_action(obs)
    f = agent.debug.get("f"); fc = None if f is None else f.copy()
    obs, r, terminated, truncated, info = env.step(act)
    v = obs["rocket_sensors"][9:12]
    if vprev is not None and fc is not None:
        rows.append((obs["simulation_time"], fc + G, (v - vprev) / 0.01, np.linalg.norm(v)))
    vprev = v.copy()
for t, ac, aa, sp in rows[::100]:
    print(f"t={t:5.1f} speed={sp:5.1f} cmd={ac.round(1)} act={aa.round(1)} err={np.linalg.norm(ac-aa):.1f}")
print("mean |err|", np.mean([np.linalg.norm(a - b) for _, a, b, _ in rows]))

"""Dev: trace one flight once per second."""
import contextlib, io, logging, sys, warnings
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import os
from BalloonPoppingGymEnv.agents.beam_agent import BeamAgent
from BalloonPoppingGymEnv.agents.gnc_agent import GNCAgent
from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters
from dev.run_seeds import parse_kw
warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
seed = int(sys.argv[1]); kw = parse_kw(sys.argv[2:])
params, given = load_scenario_parameters(1)
env = BalloonPoppingEnv(render_mode=None, parameters=params)
with contextlib.redirect_stdout(io.StringIO()):
    obs, info = env.reset(seed=seed)
agent = (BeamAgent if os.environ.get('AGENT', 'beam') == 'beam' else GNCAgent)(given, **kw); terminated = truncated = False; mind = {}
while not (terminated or truncated):
    act = agent.get_action(obs)
    obs, r, terminated, truncated, info = env.step(act)
    t = obs["simulation_time"]; s = obs["rocket_sensors"]
    if r: print(f"  POP t={t:.2f}")
    tg = agent.target
    if tg is not None and not np.isnan(s[0]):
        d = np.linalg.norm(obs["balloon_states"][tg, :3] - s[6:9])
        mind[tg] = min(mind.get(tg, 1e9), d)
    if abs(t * 100 % 100) < 0.5 and not np.isnan(s[0]):
        f = agent.debug.get("f")
        print(f"t={t:5.1f} pos={s[6:9].round(0)} vel={s[9:12].round(1)} tgt={tg} "
              f"d={d if tg is not None else -1:6.1f} thr={act['throttle']:.2f} tvc={np.round(act['tvc'],1)} aT={agent.a_thrust:.1f} f={None if f is None else f.round(1)}")
print("popped", info["popped_count"], "min dist per target", {k: round(v, 1) for k, v in mind.items()})

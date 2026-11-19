"""Dev: per-engagement closest approach, and how it ended (pop / switch / miss)."""
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
kw = parse_kw(sys.argv[2:])
params, given = load_scenario_parameters(1)
env = BalloonPoppingEnv(render_mode=None, parameters=params)
with contextlib.redirect_stdout(io.StringIO()):
    obs, info = env.reset(seed=int(sys.argv[1]))
agent = (BeamAgent if os.environ.get('AGENT', 'beam') == 'beam' else GNCAgent)(given, **kw); terminated = truncated = False
cur, start, cpa, hist = None, 0, 1e9, []
while not (terminated or truncated):
    act = agent.get_action(obs)
    tg = agent.target; t = obs["simulation_time"]
    obs, r, terminated, truncated, info = env.step(act)
    s = obs["rocket_sensors"]
    if tg != cur:
        if cur is not None:
            hist.append((cur, start, t, cpa, int(obs["balloon_status"][cur, 0])))
        cur, start, cpa = tg, t, 1e9
    if tg is not None:
        d = np.linalg.norm(obs["balloon_states"][tg, :3] - s[6:9])
        if d < cpa: cpa, v_at = d, np.linalg.norm(s[9:12])
if cur is not None: hist.append((cur, start, t, cpa, int(obs["balloon_status"][cur, 0])))
for b, t0, t1, c, st in hist:
    if t1 - t0 > 0.3: print(f"tgt {b:3d} {t0:5.1f}-{t1:5.1f}s  cpa {c:6.1f}  {'POP' if st == 2 else ''}")
print("popped", info["popped_count"])

"""Dev: step log of one engagement (seed, target index)."""
import contextlib, io, logging, os, sys, warnings
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from BalloonPoppingGymEnv.agents.beam_agent import BeamAgent
from BalloonPoppingGymEnv.agents.gnc_agent import G, quat_to_R
from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters
from dev.run_seeds import parse_kw
warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
seed, tgt_watch = int(sys.argv[1]), int(sys.argv[2]); kw = parse_kw(sys.argv[3:])
params, given = load_scenario_parameters(1)
env = BalloonPoppingEnv(render_mode=None, parameters=params)
with contextlib.redirect_stdout(io.StringIO()):
    obs, info = env.reset(seed=seed)
agent = BeamAgent(given, **kw); terminated = truncated = False; vprev = None
while not (terminated or truncated):
    act = agent.get_action(obs)
    f = agent.debug.get("f"); plan = agent.plan
    obs, r, terminated, truncated, info = env.step(act)
    s = obs["rocket_sensors"]; t = obs["simulation_time"]
    if agent.target == tgt_watch and f is not None:
        b = obs["balloon_states"][tgt_watch]
        a_act = (s[9:12] - vprev) / 0.01 if vprev is not None else np.zeros(3)
        nose = quat_to_R(info["rocket_states"][6:10])[:, 2]
        ang = np.degrees(np.arccos(np.clip(nose @ f / np.linalg.norm(f), -1, 1)))
        if round(t * 100) % 10 == 0:
            print(f"t={t:6.2f} tgo={plan[1]-t:5.2f} d={np.linalg.norm(b[:3]-s[6:9]):6.2f} |f|={np.linalg.norm(f):5.2f} aT={agent.a_thrust:5.2f} thr={act['throttle']:.2f} nose_err={ang:5.1f} acc_err={np.linalg.norm(f+G-a_act):5.2f} spd={np.linalg.norm(s[9:12]):5.1f}")
    vprev = s[9:12].copy()

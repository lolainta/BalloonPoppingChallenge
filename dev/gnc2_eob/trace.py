"""Trace the end of the burn for one seed / config: every 0.1 s of the last 3 s of the burn and the first 3 s after
it: burn left, target, reach time, eob state (tau, |f| needed), true distance to the target.
.venv/bin/python dev/gnc2_eob/trace.py --scenario 4 --seed 30000 --kw policy_path=... sl_path=... eob=1"""
import os, sys, contextlib, io, warnings, logging, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dev.run_seeds import parse_kw
from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters
from BalloonPoppingGymEnv.agents.rl3_agent import RL3Agent

ap = argparse.ArgumentParser()
ap.add_argument("--scenario", type=int, default=4); ap.add_argument("--seed", type=int, default=30000)
ap.add_argument("--kw", nargs="*", default=[]); a = ap.parse_args()
warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
params, given = load_scenario_parameters(a.scenario)
env = BalloonPoppingEnv(render_mode=None, parameters=params)
with contextlib.redirect_stdout(io.StringIO()):
    obs, _ = env.reset(seed=a.seed)
ag = RL3Agent(given, **parse_kw(a.kw))
done, k, last_tgt, pops = False, 0, None, 0
while not done:
    obs, r, te, tr, info = env.step(ag.get_action(obs)); done = te or tr; k += 1
    t = float(obs["simulation_time"])
    if info["popped_count"] > pops:
        pops = info["popped_count"]; print(f"  POP t {t:.2f}  total {pops}")
    if ag.launched_at is None:
        continue
    tb = ag.launched_at + ag.burn_time - ag.dt
    if ag.target is not None:
        last_tgt = ag.target
    if tb - 3.0 <= t <= tb + 3.0 and k % 10 == 0 and last_tgt is not None:
        rs = np.asarray(info["rocket_states"][:3], float); b = np.asarray(obs["balloon_states"][last_tgt], float)
        e = getattr(ag, "debug_eob", None) if getattr(ag, "eob_f", None) is not None else None
        f = ag.debug.get("f")
        print(f"t-tb {t - tb:+.2f}  tgt {last_tgt} st {int(obs['balloon_status'][last_tgt, 0])}  dist {np.linalg.norm(b[:3] - rs):6.2f}"
              f"  |f| {np.linalg.norm(f) if f is not None else float('nan'):5.1f}  a_thr {ag.a_thrust:5.1f}  stretch {ag.stretch}"
              + (f"  EOB tau {e[1]:.2f} |F| {e[2]:.1f}" if e else ""))
print("pops", info["popped_count"])

"""A tc2 checkpoint with search_k > 0 / the DP planner / ens raises NotImplementedError at the first decision, also with tc_off=1
(tc2_off=1 and tc_off=1: allowed); the learner drops worker messages without the "tc2" stamp (async_ppo.fits).
source dev/tc2_check/env.sh; $PY dev/tc2_check/check_guard.py TC2.npz"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dev import async_ppo as ap  # noqa: E402
from BalloonPoppingGymEnv.agents import rl3_agent  # noqa: E402
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters  # noqa: E402

assert Path(rl3_agent.__file__).resolve().is_relative_to(ROOT) and Path(ap.__file__).resolve().is_relative_to(ROOT)
_, given = load_scenario_parameters(4)
for kw in ({"search_k": 3}, {"search_k": 3, "tc_off": 1}, {"dp": 1, "tc_off": 1}, {"dp": 2, "tc_off": 1}, {"dp_shadow": 1, "tc_off": 1},
           {"ens": 1, "tc_off": 1}, {"search_k": 3, "tc_off": 1, "tc2_off": 1}):
    ag = rl3_agent.RL3Agent(given, policy_path=sys.argv[1], use_sl=0, **{k: v for k, v in kw.items() if k != "ens"})
    if "ens" in kw:
        ag.ens = [dict(ag.net)]  # (a second member, as RL3Agent's ens= would load it)
    try:
        ag.choose(None)  # (raises before it reads the observation)
        r = "no error"
    except NotImplementedError as e:
        r = f"NotImplementedError: {e}"
    except Exception as e:  # tc_off=1 tc2_off=1 gets past the guard and fails on the dummy observation
        r = f"past the guard ({type(e).__name__})"
    print(kw, "->", r)
m = {"act": np.zeros((ap.SEG, 13)), "nr": 8}
print(f"RL3_TCOND2 {ap.TC2}: fits(unstamped 13-wide) {ap.fits(m, 13)}, fits(stamped) {ap.fits({**m, 'tc2': 1}, 13)}")

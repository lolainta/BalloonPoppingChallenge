"""A fast checkpoint with search_k > 0 / the DP planner / ens raises NotImplementedError at the first decision, also with tc_off=1 tc2_off=1
(+ fast_off=1: allowed); an RL3_FAST learner takes only 14-column messages stamped "fast" (async_ppo.fits).
source dev/fast_check/env.sh; RL3_FAST=10 $PY dev/fast_check/check_guard.py FAST.npz"""
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
off = {"tc_off": 1, "tc2_off": 1}
for kw in ({"search_k": 3, **off}, {"dp": 1, **off}, {"dp": 2, **off}, {"dp_shadow": 1, **off}, {"ens": 1, **off},
           {"search_k": 3, **off, "fast_off": 1}):
    ag = rl3_agent.RL3Agent(given, policy_path=sys.argv[1], use_sl=0, **{k: v for k, v in kw.items() if k != "ens"})
    if "ens" in kw:
        ag.ens = [dict(ag.net)]
    try:
        ag.choose(None)
        r = "no error"
    except NotImplementedError as e:
        r = f"NotImplementedError: {e}"
    except Exception as e:  # past the guard: fails on the dummy observation
        r = f"past the guard ({type(e).__name__})"
    print(kw, "->", r)
m = {"act": np.zeros((ap.SEG, 14)), "nr": 8, "tc2": 1}
print(f"RL3_FAST {ap.FAST}: fits(13-wide pre-change, stamped nr/tc2) {ap.fits({**m, 'act': np.zeros((ap.SEG, 13))}, 14)}, "
      f"fits(14-wide without 'fast') {ap.fits(m, 14)}, fits(14-wide stamped) {ap.fits({**m, 'fast': 1}, 14)}")

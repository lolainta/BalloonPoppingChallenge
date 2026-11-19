"""A tc checkpoint with search_k > 0 / the DP planner raises NotImplementedError at the first decision (tc_off=1: allowed).
python dev/tcond_check/check_guard.py TC.npz"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from BalloonPoppingGymEnv.agents import rl3_agent  # noqa: E402
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters  # noqa: E402

assert Path(rl3_agent.__file__).resolve().is_relative_to(ROOT), rl3_agent.__file__
_, given = load_scenario_parameters(4)
for kw in ({"search_k": 3}, {"dp": 1}, {"dp": 2}, {"dp_shadow": 1}, {"search_k": 3, "tc_off": 1}):
    ag = rl3_agent.RL3Agent(given, policy_path=sys.argv[1], use_sl=0, **kw)
    try:
        ag.choose(None)  # (raises before it reads the observation)
        r = "no error"
    except NotImplementedError as e:
        r = f"NotImplementedError: {e}"
    except Exception as e:  # tc_off=1 gets past the guard and fails on the dummy observation
        r = f"past the guard ({type(e).__name__})"
    print(kw, "->", r)

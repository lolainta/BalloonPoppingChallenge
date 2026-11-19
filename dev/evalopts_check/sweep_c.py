"""C: dev/sweep_gnc.py with every eval option as a --cfg entry (paired, one reset per seed), after asserting that
this tree's agents are the ones imported (sweep_gnc puts its own repo root first on sys.path, workers included).
EVALOPTS_MINE=1 nice -n 10 .venv/bin/python dev/evalopts_check/sweep_c.py --seeds 10000-10001"""
import os, sys

root = os.path.realpath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, root)
from BalloonPoppingGymEnv.agents import gnc_agent, rl2_agent, rl3_agent  # noqa: E402
for m in (gnc_agent, rl2_agent, rl3_agent):
    assert os.path.realpath(m.__file__).startswith(root + "/"), m.__file__
R = "./runs/s4res2"
ens = f"ens={R}/rl3_policy_116438400.npz,{R}/rl3_policy_116179200.npz"
cfgs = ["", "ho_keep=1.0", "redecide=1", "redecide=2", "ho_T=1.0", "eob_pick=1", "eob_align=20", ens,
        f"ho_keep=1.0 redecide=2 ho_T=1.0 eob_pick=1 eob_align=20 {ens}"]
os.execv(sys.executable, [sys.executable, f"{root}/dev/sweep_gnc.py", "--scenario", "4", *sys.argv[1:], "--base",
                          f"policy_path={R}/rl3_policy_116668800.npz", f"sl_path={R}/rl2_sl_116668800.npz", "P.max_wall_time=36000", "--cfg", *cfgs])

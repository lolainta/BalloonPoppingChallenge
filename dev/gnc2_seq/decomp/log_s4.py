"""Scenario-4 free flights of the current policy, logged like dev/gnc2_log.py ours(mode="free") (same npz layout, subset).

.venv/bin/python dev/gnc2_seq/decomp/log_s4.py 36000 36039 [procs]
-> dev/gnc2_seq/decomp/data/s4free_<seed>.npz : t, R (T,13), B (T,100,6), S (T,100), TGT, launch, pops (K,2) [t, b], seed
"""
import contextlib, io, os, sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).resolve().parent / "data"
CK = "runs/s4bw4/rl3_policy_26342400.npz"
SL = "runs/s4bw4/rl2_sl_26342400.npz"
KW = dict(eob=2, eob_tau=3, eob_T=4)


def fly(seed):
    import warnings; warnings.filterwarnings("ignore")
    from BalloonPoppingGymEnv.agents.rl3_agent import RL3Agent
    from dev import rl2_env
    os.chdir(ROOT)
    params, given = rl2_env.load_scenario_parameters(4)
    env = rl2_env.BalloonPoppingEnv(render_mode=None, parameters=params)
    with contextlib.redirect_stdout(io.StringIO()):
        obs, _ = env.reset(seed=seed)
    ag = RL3Agent(given, policy_path=CK, sl_path=SL, **KW)
    L = {k: [] for k in "t R B S TGT".split()}
    done = False
    while not done:
        a = ag.get_action(obs)
        obs, r, term, trunc, inf = env.step(a)
        done = term or trunc
        dbg = ag.debug or {}
        L["t"].append(float(obs["simulation_time"])); L["R"].append(np.asarray(inf["rocket_states"], float))
        L["B"].append(obs["balloon_states"].astype(np.float32)); L["S"].append(obs["balloon_status"][:, 0].astype(np.int8))
        L["TGT"].append(-1 if dbg.get("target") is None else int(dbg["target"]))
    S = np.array(L["S"]); t = np.array(L["t"])
    pops = sorted((t[int(np.argmax(S[:, b] == 2))], b) for b in range(S.shape[1]) if (S[:, b] == 2).any())
    R = np.array(L["R"]); live = ~np.isnan(R[:, 0])
    np.savez_compressed(OUT / f"s4free_{seed}.npz", **{k: np.array(v) for k, v in L.items()},
                        launch=float(t[int(np.argmax(live))]), pops=np.array(pops, float).reshape(-1, 2), seed=seed,
                        burn_time=ag.burn_time)
    return seed, len(pops)


if __name__ == "__main__":
    import multiprocessing as mp
    a, b = int(sys.argv[1]), int(sys.argv[2]); procs = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    OUT.mkdir(parents=True, exist_ok=True)
    with mp.get_context("spawn").Pool(min(procs, 12)) as p:
        for s, n in p.imap_unordered(fly, range(a, b + 1)):
            print(s, n, flush=True)

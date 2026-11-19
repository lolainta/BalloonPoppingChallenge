"""Log a strict-order flight with a chaining mode in the data/gnc2 format (t, R, B, S, launch, pops) -> logs/ours_<tag>_<name>.npz,
so hops.py (HOPS_DIR=logs) can measure its hop geometry. Usage: chain_log.py MODE R_OK W TAG  (MODE 3 = ZEV with W = kappa)"""
import contextlib, io, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import chain_sim as C  # noqa: E402
OUT = Path(__file__).parent / "logs"


def run(name, mode, r_ok, w, tag):
    import warnings; warnings.filterwarnings("ignore")
    cls = C.ChainTeacher
    if mode == 3:
        import chain_sim3
        cls = chain_sim3.ZEVTeacher
    info = C.rt.replay_info(str(C.ROOT.parent / "lb_replays" / f"{name}.json"))
    params, given = C.rl2_env.load_scenario_parameters(1)
    env = C.rl2_env.BalloonPoppingEnv(render_mode=None, parameters=params)
    with contextlib.redirect_stdout(io.StringIO()):
        obs, _ = env.reset(seed=info["seed"])
    ag = cls(given, info, mode=mode, r_ok=r_ok, w_turn=w, policy_path=C.CK, sl_path=C.CK.replace("rl3_policy", "rl2_sl"))
    L = {k: [] for k in "t R B S".split()}
    done = False
    while not done:
        obs, r, term, trunc, inf = env.step(ag.get_action(obs))
        done = term or trunc
        L["t"].append(float(obs["simulation_time"])); L["R"].append(np.asarray(inf["rocket_states"], float))
        L["B"].append(obs["balloon_states"].astype(np.float32)); L["S"].append(obs["balloon_status"][:, 0].astype(np.int8))
    S, t, R = np.array(L["S"]), np.array(L["t"]), np.array(L["R"])
    pops = sorted((t[int(np.argmax(S[:, b] == 2))], b) for b in range(S.shape[1]) if (S[:, b] == 2).any())
    live = ~np.isnan(R[:, 0])
    np.savez_compressed(OUT / f"ours_{tag}_{name}.npz", t=t, R=R, B=np.array(L["B"]), S=S, launch=float(t[int(np.argmax(live))]),
                        pops=np.array(pops, float), seed=info["seed"])
    return f"{name} {len(pops)}"


if __name__ == "__main__":
    import multiprocessing as mp
    mode, r_ok, w, tag = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
    OUT.mkdir(exist_ok=True)
    names = [n for n in C.NAMES if n not in ("13_Apex", "16_SpaceBerry")]
    with mp.get_context("spawn").Pool(4) as p:
        print(p.starmap(run, [(n, mode, r_ok, w, tag) for n in names]))

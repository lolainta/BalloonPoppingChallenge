"""Behaviour-cloning demos for RL2: fly a teacher agent and, at every RL2 decision point, record the RL2
observation, the teacher's target and the nearest stretch level, plus pops-to-go (critic warm start).

AGENT=beam .venv/bin/python dev/bc_collect.py --seeds 30000-30039 --out data/bc/bc_0.npz --kw leg_dt=0.1 keep_bias=100 replan_every=1
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse, contextlib, io, logging, multiprocessing as mp, sys, warnings
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from BalloonPoppingGymEnv.agents.beam_agent import BeamAgent  # noqa: E402
from BalloonPoppingGymEnv.agents.gnc_agent import GNCAgent  # noqa: E402
from BalloonPoppingGymEnv.agents.rl2_agent import DECIDE, STRETCH, RL2Agent, reach_time  # noqa: E402
from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv  # noqa: E402
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters  # noqa: E402


def run(args):
    seed, kw = args
    warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
    params, given = load_scenario_parameters(1)
    try:
        env = BalloonPoppingEnv(render_mode=None, parameters=params)
        with contextlib.redirect_stdout(io.StringIO()):
            obs, _ = env.reset(seed=seed)
    except Exception:
        return seed, -1, [], [], [], []
    ag = {"beam": BeamAgent, "gnc": GNCAgent}[os.environ.get("AGENT", "beam")](given, **kw)
    ag.cmd, ag.sl, since, n_pop = None, None, DECIDE, 0
    X, A, S, T = [], [], [], []
    done = False
    while not done:
        act = ag.get_action(obs)
        t = float(obs["simulation_time"])
        st = obs["balloon_status"][:, 0]
        if ag.p_hat is not None and ag._burning(t) and ag.target is not None and st[ag.target] == 1:
            popped = int((st == 2).sum())
            if ag.cmd is None or since >= DECIDE or popped > n_pop or st[ag.cmd[0]] != 1:
                b = obs["balloon_states"][ag.target]
                t1 = reach_time(ag.p_hat, ag.v_hat, b[None, :3], b[None, 3:], ag.a_thrust - ag.margin, ag.t_grid)[0]
                if isinstance(ag, BeamAgent) and ag.plan is not None and np.isfinite(t1):
                    ratio = (ag.plan[1] - t) / t1
                else:
                    ratio = getattr(ag, "stretch", 1.3)
                s = int(np.argmin(np.abs(np.array(STRETCH) - ratio)))
                X.append(RL2Agent.features(ag, obs)); A.append(ag.target); S.append(s); T.append(popped)
                ag.cmd, since, n_pop = (ag.target, STRETCH[s]), 0, popped
        since += 1
        obs, r, te, tr, info = env.step(act)
        done = te or tr
    final = int(info["popped_count"])
    return seed, final, X, A, S, [final - p for p in T]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--kw", nargs="*", default=[])
    a = ap.parse_args()
    lo, _, hi = a.seeds.partition("-")
    kw = {k: float(v) for k, v in (it.split("=") for it in a.kw)}
    with mp.Pool(a.procs) as pool:
        res = pool.map(run, [(s, kw) for s in range(int(lo), int(hi or lo) + 1)])
    ok = [r for r in res if r[1] >= 0 and r[2]]
    X = np.concatenate([np.array(r[2], np.float32) for r in ok])
    np.savez_compressed(a.out, X=X, A=np.concatenate([r[3] for r in ok]), S=np.concatenate([r[4] for r in ok]),
                        R=np.concatenate([r[5] for r in ok]).astype(np.float32), pops=np.array([(r[0], r[1]) for r in res]))
    print(f"{len(ok)} episodes, {len(X)} samples, mean pops {np.mean([r[1] for r in ok]):.2f}")

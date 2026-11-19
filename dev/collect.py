"""Collect leg labels for a learned reachability model. Not used for evaluation.

Each leg = one committed (state, target, planned time) -> (hit, actual time, v at end).
The agent sticks to a target until hit / arrival time passed / burnout; eps of commits
pick a random balloon so labels also cover legs the heuristic would never choose.

uv run python dev/collect.py --seeds 0-95 --out legs_0.npz [--eps 0.3] [--procs 8]
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")  # ponytail: one BLAS thread per worker, else 24 procs x 24 threads
import argparse
import contextlib
import io
import logging
import multiprocessing as mp
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from BalloonPoppingGymEnv.agents.beam_agent import BeamAgent  # noqa: E402
from BalloonPoppingGymEnv.agents.gnc_agent import quat_to_R  # noqa: E402
from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv  # noqa: E402
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters  # noqa: E402


class CollectAgent(BeamAgent):
    def __init__(self, given, eps, rng, **kw):
        super().__init__(given, **kw)
        self.eps, self.rng = eps, rng
        self.cur, self.X, self.Y = None, [], []

    def _close(self, t, status):
        if self.cur is None:
            return
        x, t0, tgt = self.cur
        self.X.append(x)
        self.Y.append(np.r_[status[tgt] == 2, t - t0, self.v_hat])
        self.cur = None

    def _plan(self, t, p, v, obs):
        status = obs["balloon_status"][:, 0]
        if self.plan is not None and status[self.plan[0]] == 1 and t <= self.plan[1]:
            return self.plan  # commit: no replanning mid-leg
        self._close(t, status)
        plan = super()._plan(t, p, v, obs)
        alive = np.flatnonzero(status == 1)
        if alive.size and (plan is None or self.rng.random() < self.eps):
            tgt = int(self.rng.choice(alive))
            b = obs["balloon_states"][tgt]
            tb = np.array([self.launched_at + self.burn_time - t])
            tmin = self._legs(p[None], v[None], tb, b[None, None, :3], b[None, None, 3:], quat_to_R(self.q)[:, 2][None])[0, 0]
            # ponytail: uniform T up to max_leg so later-leg labels are not all "analytic tmin -> miss"
            lo = min(tmin, 8.0) if np.isfinite(tmin) else 1.0
            tleg = self.rng.uniform(lo, self.leg_grid[-1])
            plan = (tgt, t + tleg)
        if plan is None:
            return None
        tgt, t_arr = plan
        b = obs["balloon_states"][tgt]
        burn_left = self.launched_at + self.burn_time - t
        x = np.r_[b[:3] - p, b[3:] - v, v, quat_to_R(self.q)[:, 2], burn_left, t_arr - t]
        self.cur = (x, t, tgt)
        return plan


def run(args):
    seed, scenario, eps, kw = args
    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)
    params, given = load_scenario_parameters(scenario)
    env = BalloonPoppingEnv(render_mode=None, parameters=params)
    agent = CollectAgent(given, eps, np.random.default_rng(seed), **kw)
    done, info = False, {"popped_count": 0}
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            obs, info = env.reset(seed=seed)
        while not done:
            obs, r, term, trunc, info = env.step(agent.get_action(obs))
            done = term or trunc
    except Exception as e:  # ponytail: env-side crash (e.g. negative mass); keep the legs closed so far
        print(f"seed {seed}: {e}", file=sys.stderr)
        if agent.cur is None and not agent.X:
            return seed, -1, np.zeros((0, 14)), np.zeros((0, 5))  # reset failed: unusable seed
    agent._close(float(obs["simulation_time"]), obs["balloon_status"][:, 0])  # coast hit after burnout
    return seed, info["popped_count"], np.array(agent.X), np.array(agent.Y)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-7")
    ap.add_argument("--scenario", type=int, default=1)
    ap.add_argument("--eps", type=float, default=0.3)
    ap.add_argument("--procs", type=int, default=mp.cpu_count())
    ap.add_argument("--out", required=True)
    ap.add_argument("--kw", nargs="*", default=[])
    a = ap.parse_args()
    lo, _, hi = a.seeds.partition("-")
    seeds = list(range(int(lo), int(hi or lo) + 1))
    kw = {k: float(v) for k, v in (it.split("=") for it in a.kw)}
    with mp.Pool(min(len(seeds), a.procs)) as pool:
        res = pool.map(run, [(s, a.scenario, a.eps, kw) for s in seeds])
    X = np.concatenate([r[2] for r in res if len(r[2])])
    Y = np.concatenate([r[3] for r in res if len(r[3])])
    S = np.concatenate([np.full(len(r[2]), r[0]) for r in res])
    np.savez(a.out, X=X, Y=Y, seed=S, pops=np.array([(r[0], r[1]) for r in res]))
    print(f"{len(seeds)} seeds  legs {len(X)}  hit rate {Y[:, 0].mean():.2f}  mean pops {np.mean([r[1] for r in res]):.2f}")

"""Pops vs dv on real scenario-4 flights (the verifier probe, 2026-10-02): a zero-grown 8-output npz; dv forced at every decision to
DVMAG * (1, -0.6, 0.8) / |.| (DVMAG >= 0), or (DVMAG < 0) to res_vbound tanh(u[5:8] + SIG N(0, 1)) = the worker's sampling of
a zero head with std SIG (env, default 0.3; rng stream |10 DVMAG|).
PYTHONPATH=<tree> python pops_vs_dv.py <tree> POLICY.npz SL.npz SEEDS(,) DVMAGS(,)"""
import os
SIG = float(os.environ.get("SIG", 0.3))
import contextlib, io, logging, sys, warnings, json, multiprocessing as mp
from pathlib import Path
import numpy as np
W = Path(sys.argv[1]); sys.path.insert(0, str(W))
def run(args):
    pol, sl, seed, dvmag, branch_only = args
    warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
    from BalloonPoppingGymEnv.agents import gnc_agent, rl3_agent
    assert Path(gnc_agent.__file__).resolve().is_relative_to(W)
    from BalloonPoppingGymEnv.agents.gnc_agent import GNCAgent
    from BalloonPoppingGymEnv.agents.rl2_sl import load_sl
    from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv
    from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters
    st = {"coast": 0, "n2": 0, "arrv": [], "ratio": []}
    g0, a0 = GNCAgent._guidance, GNCAgent._arrv
    def g(self, p, v, bp, bv, t_go):
        t = max(t_go * self.stretch * self.s_mul, self.t_go_min)
        self._br = "coast" if (self.coast and t > self.burn_left) else "n2"
        out = g0(self, p, v, bp, bv, t_go); self._an2 = np.linalg.norm(out); return out
    def ar(self, t_go):
        da = a0(self, t_go); st[self._br] += 1; st["arrv"].append(np.linalg.norm(da)); st["ratio"].append(np.linalg.norm(da) / max(self.a_thrust, 1e-9))
        return da
    GNCAgent._guidance, GNCAgent._arrv = g, ar
    s0 = rl3_agent.RL3Agent.set_res
    d = np.array([1.0, -0.6, 0.8]); d /= np.linalg.norm(d)
    def sr(self, u):
        s0(self, u)
        if self.res_vbound and len(u) > 5:
            if dvmag < 0: self.dv = self.res_vbound * np.tanh(np.asarray(u[5:8]) + SIG * rng.standard_normal(3))
            else: self.dv = dvmag * d
    rl3_agent.RL3Agent.set_res = sr
    rng = np.random.default_rng(abs(int(dvmag * 10)))
    params, given = load_scenario_parameters(4)
    env = BalloonPoppingEnv(render_mode=None, parameters=params)
    with contextlib.redirect_stdout(io.StringIO()):
        obs, _ = env.reset(seed=seed)
    ag = rl3_agent.RL3Agent(given, policy_path=pol); ag.sl = load_sl(sl)
    done = False
    while not done:
        obs, r, te, tr, info = env.step(ag.get_action(obs)); done = te or tr
    a = np.array(st["arrv"] or [0]); rr = np.array(st["ratio"] or [0])
    return dvmag, seed, int(info["popped_count"]), st["coast"], st["n2"], float(a.mean()), float((rr >= ag.vcap - 1e-9).mean())
if __name__ == "__main__":
    pol, sl = sys.argv[2], sys.argv[3]; seeds = [int(s) for s in sys.argv[4].split(",")]; mags = [float(m) for m in sys.argv[5].split(",")]
    jobs = [(pol, sl, s, m, 0) for s in seeds for m in mags]
    with mp.Pool(min(6, len(jobs))) as p:
        for r in p.map(run, jobs):
            print("dv %.1f seed %d pops %d  arrv steps coast-branch %d n2-branch %d  mean|da| %.2f  capped frac %.2f" % r, flush=True)

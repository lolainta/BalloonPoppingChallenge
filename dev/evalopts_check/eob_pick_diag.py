"""Why does eob_pick (not) change the tau: per _eob_pick call, the number of feasible tau and the closest predicted
post-hit approach to another live balloon (any tau / feasible tau), 5-output checkpoint, one seed.
PYTHONPATH=<tree> .venv/bin/python dev/evalopts_check/eob_pick_diag.py SEED [k=v ...]"""
import contextlib, io, os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
from BalloonPoppingGymEnv.agents import gnc_agent, rl3_agent
from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters

root = os.path.realpath(os.path.join(os.path.dirname(__file__), "../.."))
for m in (gnc_agent, rl3_agent):
    assert os.path.realpath(m.__file__).startswith(root + "/"), m.__file__
R = "./runs/s4res2"
G = gnc_agent.G
rec = []
orig = gnc_agent.GNCAgent._eob_pick


def spy(self, obs, v, T, tau, B, F, k, horizon=1.5, ds=0.05):
    st, bs = obs["balloon_status"][:, 0], obs["balloon_states"]
    oth = np.flatnonzero((st == 1) & np.isfinite(bs).all(1)); oth = oth[oth != self.target]
    fn = np.linalg.norm(F, axis=1); feas = fn <= self.a_thrust * self.thr_hi
    s = np.arange(0.0, horizon + 1e-9, ds)[None, :, None]
    vh = v + (F + G) * T + G * tau[:, None]
    path = B[:, None] + vh[:, None] * s + 0.5 * G * s**2
    ts = (T + tau)[:, None, None] + s
    ob = bs[oth, :3] + bs[oth, 3:] * ts[..., None]
    if self.bwind:
        ob = ob + self._bdrift(bs[oth, 2], bs[oth, 5], ts, self.aim_prof)
    d = np.linalg.norm(ob - path[:, :, None], axis=-1).min((1, 2)) if oth.size else np.full(len(tau), np.inf)
    j = orig(self, obs, v, T, tau, B, F, k, horizon, ds)
    rec.append((round(T, 2), self.target, int(feas.sum()), float(fn.min()), float(self.a_thrust * self.thr_hi),
                float(d.min()), float(d[feas].min()) if feas.any() else np.inf, k, j))
    return j


gnc_agent.GNCAgent._eob_pick = spy
seed, *rest = sys.argv[1:]
kw = {a.split("=")[0]: float(a.split("=")[1]) for a in rest}
params, given = load_scenario_parameters(4)
env = BalloonPoppingEnv(render_mode=None, parameters=params)
with contextlib.redirect_stdout(io.StringIO()):
    obs, _ = env.reset(seed=int(seed))
ag = rl3_agent.RL3Agent(given, policy_path=f"{R}/rl3_policy_116668800.npz", sl_path=f"{R}/rl2_sl_116668800.npz", eob_pick=1, **kw)
done = False
while not done:
    obs, r, te, tr, info = env.step(ag.get_action(obs)); done = te or tr
print("popped", info["popped_count"], "counts", ag.opt_n)
print("T_left target n_feasible min|F| thrust_max  min_dist_any_tau  min_dist_feasible_tau  k_argmin k_pick")
for r in rec[:: max(1, len(rec) // 25)]:
    print(*[f"{x:.2f}" if isinstance(x, float) else x for x in r])

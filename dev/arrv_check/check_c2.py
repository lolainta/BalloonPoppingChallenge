"""C(ii): one deterministic scenario-4 flight with a non-zero dv checkpoint; on every step a shadow copy of the GNC (deepcopy,
dv = 0 and no deviation state) runs the same step. Guidance steps (burning, a target, no eob / coast-planner override):
a_cmd(real) - a_cmd(shadow) must equal gnc_agent._arrv's term recomputed here from the state before the step (t = max(t_go
stretch s_mul, t_go_min), tf = max(t, 0.5), N = nav_N n_mul, (dv_p, dv_v) reset on a new target, Z = -(dv_p + dv_v t):
((6 - N) Z / tf - 2 (dv - dv_v)) / tf * clip((t - 0.3) / 0.4, 0, 1), capped at vcap * a_thrust; 0 in the coast branch t > burn
left), and the deviation state must advance by (N Z / t^2 + term) over dt. Every other step (pad, after burnout, no target,
eob / planner): f(real) = f(shadow).
python check_c2.py POLICY.npz SL.npz SEED   (PYTHONPATH = this tree)"""
import contextlib, copy, io, logging, os, sys, warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
from BalloonPoppingGymEnv.agents import gnc_agent, rl3_agent  # noqa: E402
from BalloonPoppingGymEnv.agents.gnc_agent import GNCAgent  # noqa: E402
from BalloonPoppingGymEnv.agents.rl2_sl import load_sl  # noqa: E402
from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv  # noqa: E402
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters  # noqa: E402

assert Path(gnc_agent.__file__).resolve().is_relative_to(ROOT) and Path(rl3_agent.__file__).resolve().is_relative_to(ROOT)
pol, sl, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])

sat0, spend0, act0 = GNCAgent._saturate, GNCAgent._spend_excess, GNCAgent._gnc_action


def sat(self, a):
    self._sat_in = a.copy()
    return sat0(self, a)


def spend(self, f, v):
    self._acmd = self._sat_in  # the main guidance call's a_cmd (_ho_blend also saturates, earlier)
    return spend0(self, f, v)


def expected(ag, st0):
    dp, dv_, tgt = st0
    t = max(ag.t_go_last[1] * ag.stretch * ag.s_mul, ag.t_go_min)
    if ag.target != tgt:
        dp, dv_ = np.zeros(3), np.zeros(3)
    if ag.coast and t > ag.burn_left:
        return np.zeros(3), t, False, (dp, dv_), True
    N, tf = ag.nav_N * ag.n_mul, max(t, 0.5)
    z = -(dp + dv_ * t)
    da = ((6.0 - N) * z / tf - 2.0 * (ag.dv - dv_)) / tf * min(max((t - 0.3) / 0.4, 0.0), 1.0)
    n, cap = np.linalg.norm(da), ag.vcap * ag.a_thrust
    da = da * cap / n if n > cap else da
    a = N * z / t**2 + da
    return da, t, n > cap, (dp + dv_ * ag.dt + 0.5 * a * ag.dt**2, dv_ + a * ag.dt), False


stats = {k: 0 for k in ("pad_or_burnout", "no_target", "override", "guidance", "ground_clamp", "faded", "capped", "dv_zero", "coast_branch", "new_target")}
err = {"guidance_max_abs": 0.0, "other_f_max_abs": 0.0, "state_max_abs": 0.0}
da_n = []


def act(self, obs):
    if getattr(self, "_shadow", False):
        return act0(self, obs)
    sh = copy.deepcopy(self); sh._shadow = True; sh.dv = sh.dv_p = sh.dv_v = np.zeros(3); sh._acmd = None; self._acmd = None
    st0 = (self.dv_p.copy(), self.dv_v.copy(), self.dv_tgt)
    a_sh = act0(sh, obs)
    a = act0(self, obs)
    burning = "f" in self.debug
    if not burning:
        stats["pad_or_burnout"] += 1
        assert all(np.array_equal(np.asarray(a[k]), np.asarray(a_sh[k])) for k in a), "pad / burnout step differs"
        return a
    df = float(np.abs(self.debug["f"] - sh.debug["f"]).max())
    if self.target is None or getattr(self, "eob_f", None) is not None or self.plan_on:
        stats["no_target" if self.target is None else "override"] += 1
        err["other_f_max_abs"] = max(err["other_f_max_abs"], df)
        return a
    stats["guidance"] += 1
    if not self.dv.any():
        stats["dv_zero"] += 1
    e, t, capped, st1, coast = expected(self, st0)
    stats["coast_branch"] += coast; stats["new_target"] += self.target != st0[2] and (self.dv.any() or st0[0].any())
    if self.dv.any() or st0[0].any() or st0[1].any():  # (else _arrv is not called: state unchanged)
        err["state_max_abs"] = max(err["state_max_abs"], float(np.abs(np.r_[self.dv_p - st1[0], self.dv_v - st1[1]]).max()))
    alt = self._acmd is not None and (self.p_hat[2] - self.elev < self.min_alt and self.v_hat[2] < 0)
    if alt:  # ground avoidance may clamp a_cmd[2] after the term: compare x / y only
        stats["ground_clamp"] += 1
        d = (self._acmd - sh._acmd - e)[:2]
    else:
        d = self._acmd - sh._acmd - e
    err["guidance_max_abs"] = max(err["guidance_max_abs"], float(np.abs(d).max()))
    stats["faded"] += t < 0.7; stats["capped"] += capped
    da_n.append((np.linalg.norm(e), np.linalg.norm(self.dv), t))
    return a


GNCAgent._saturate, GNCAgent._spend_excess, GNCAgent._gnc_action = sat, spend, act
params, given = load_scenario_parameters(4)
env = BalloonPoppingEnv(render_mode=None, parameters=params)
with contextlib.redirect_stdout(io.StringIO()):
    obs, _ = env.reset(seed=seed)
agent = rl3_agent.RL3Agent(given, policy_path=pol); agent.sl = load_sl(sl)
done, n = False, 0
while not done:
    obs, r, te, tr, info = env.step(agent.get_action(obs)); n += 1; done = te or tr
print(f"seed {seed} steps {n} popped {info['popped_count']} res_vbound {agent.res_vbound} vcap {agent.vcap}")
print("steps:", stats)
print("max |a_cmd(real) - a_cmd(shadow) - expected| on guidance steps: %.3e;  max |deviation state - expected|: %.3e" % (err["guidance_max_abs"], err["state_max_abs"]))
print("max |f(real) - f(shadow)| on no-target / eob / planner steps: %.3e   (pad / burnout steps: actions identical)" % err["other_f_max_abs"])
da_n = np.array(da_n)
print("on guidance steps: |da| mean %.3f max %.3f (cap %.3f) m/s^2;  |dv| median %.2f max %.2f m/s;  t median %.2f min %.2f s" % (da_n[:, 0].mean(), da_n[:, 0].max(), agent.vcap * agent.a_thrust, np.median(da_n[:, 1]), da_n[:, 1].max(), np.median(da_n[:, 2]), da_n[:, 2].min()))

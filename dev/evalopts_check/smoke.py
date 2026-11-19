"""Seconds-long synthetic forcing tests of the GNC options (no env reset): eob_pick picks a non-argmin tau when only
that tau's post-hit coast passes a balloon; eob_align turns the held force onto the velocity; _ho_blend keeps the old
target (lambda < 1) when it is close ahead.
PYTHONPATH=<tree> .venv/bin/python dev/evalopts_check/smoke.py"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
from BalloonPoppingGymEnv.agents import gnc_agent
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters

root = os.path.realpath(os.path.join(os.path.dirname(__file__), "../.."))
assert os.path.realpath(gnc_agent.__file__).startswith(root + "/"), gnc_agent.__file__
G = gnc_agent.G
_, given = load_scenario_parameters(4)


def agent(**kw):
    ag = gnc_agent.GNCAgent(given, eob=2, eob_T=4.0, eob_tau=3.0, **kw)
    ag.launched_at, ag.burn_left, ag.stretch = 0.0, 1.0, 1.0
    return ag


el = agent().elev
p, v = np.array([0.0, 0.0, el + 60]), np.array([0.0, 12.0, 25.0])
bs = np.zeros((3, 6)); bs[:, 5] = 1.0
bs[0, :3] = [8.0, 40.0, el + 95]  # the target: not reachable by burnout (1 s), hit tau after it
bs[1, :3] = [200.0, 0.0, el + 50]; bs[2, :3] = [-200.0, 0.0, el + 50]  # far away
obs = {"balloon_status": np.ones((3, 1)), "balloon_states": bs.copy()}

# eob_pick: the plain mode's tau, then a balloon put on the coast path of another feasible tau
a0 = agent(); a0.target = 0
f0 = a0._eob(p, v, obs, 5.0, np.zeros(3)); T0, tau0, F0 = a0.debug_eob
assert a0.eob_f is not None, "eob mode not engaged"
tau = np.arange(0.05, 3.0 + 1e-9, 0.05); tt = 1.0 + tau
F = (bs[0, :3] + bs[0, 3:] * tt[:, None] - p - v * tt[:, None] - 0.5 * G * tau[:, None] ** 2) / (0.5 + tau)[:, None] - G
fn = np.linalg.norm(F, axis=1); feas = np.flatnonzero(fn <= a0.a_thrust * a0.thr_hi)
k0 = int(np.argmin(fn)); j = int(feas[np.argmax(np.abs(feas - k0))])  # the feasible tau farthest from the argmin
vh = v + (F[j] + G) + G * tau[j]; s = 0.6
B = bs[0, :3] + bs[0, 3:] * tt[j]
bs2 = bs.copy(); bs2[1, :3] = B + vh * s + 0.5 * G * s**2 - bs[1, 3:] * (tt[j] + s)  # there at hit + 0.6 s
obs2 = {"balloon_status": np.ones((3, 1)), "balloon_states": bs2}
for pick in (0, 1):
    a = agent(eob_pick=pick, eob_pick_r=0.3); a.target = 0
    f = a._eob(p, v, obs2, 5.0, np.zeros(3))
    print(f"eob_pick={pick}: tau {a.debug_eob[1]:.2f} |F| {a.debug_eob[2]:.2f} (argmin tau {tau[k0]:.2f}, planted tau "
          f"{tau[j]:.2f}, feasible {len(feas)}/{len(tau)})  force {np.round(f, 2)}  counts {a.opt_n}")
assert a.opt_n.get("eob_pick") == 1 and abs(a.debug_eob[1] - tau[j]) < 0.051, "eob_pick did not move to the planted tau"

# eob_align: hold phase (burn left <= eob_hold) with the held force 10 deg off the velocity
for al in (0, 20):
    a = agent(eob_align=al); a.target, a.burn_left, a.eob_tgt = 0, 0.2, 0
    c, sn = np.cos(np.radians(10)), np.sin(np.radians(10))
    a.eob_f = 14.0 * (c * v / np.linalg.norm(v) + sn * np.array([1.0, 0.0, 0.0]))
    f = a._eob(p, v, obs, 5.0, np.zeros(3))
    ang = np.degrees(np.arccos(f @ v / np.linalg.norm(f) / np.linalg.norm(v)))
    print(f"eob_align={al}: |f| {np.linalg.norm(f):.2f} angle to v {ang:.1f} deg  counts {a.opt_n}")
assert ang < 1e-3 and abs(np.linalg.norm(f) - 14.0) < 1e-9

# ho_keep: old target 15 m ahead on the velocity, new one behind
a = agent(ho_keep=1.0); a.target, a.ho_old = 1, 0
bs3 = bs.copy(); bs3[0, :3] = p + v / np.linalg.norm(v) * 15.0; bs3[0, 3:] = 0.0; bs3[1, :3] = p + [-40.0, -30.0, -20.0]; bs3[1, 3:] = 0
obs3 = {"balloon_status": np.ones((3, 1)), "balloon_states": bs3}
to = a._reach_time(p, v, bs3[1:2, :3], bs3[1:2, 3:])[0]
a_new = a._guidance(p, v, bs3[1, :3], bs3[1, 3:], to)
out = a._ho_blend(p, v, obs3, a_new)
print(f"ho_keep: a_new {np.round(a_new, 2)} -> blended {np.round(out, 2)}  ho_old {a.ho_old}  counts {a.opt_n}")
assert a.opt_n.get("ho_keep") == 1 and not np.allclose(out, a_new)
print("smoke OK")

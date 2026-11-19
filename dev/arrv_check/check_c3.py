"""C(iii): the math of gnc_agent._arrv, numerically (no env).
1) ZEM/ZEV law a = 6 ZEM / t^2 - 2 ZEV / t, ZEV = v_f - (v + G t), ZEM = b + bv t - (p + v t + G t^2 / 2), a = the thrust
   acceleration (gravity separate): with v_f = the N = 2 arrival velocity v + G t + 2 ZEM / t it equals the N = 2 law
   2 ZEM / t^2, and with v_f + dv it equals that - 2 dv / t (random states, max abs error).
2) point mass, constant (moving) target, gravity, no drag, t_go = the exact time left (stretch 1), dt 0.01 s, flown with:
   N=2 | ZEM/ZEV with v_f fixed at the start (= v_f(N=2) + dv) | the held-dv law of _arrv (pure; + t floor 0.5 s + fade; + cap)
   -> miss and arrival velocity - the N = 2 one, per dv.
python check_c3.py"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from BalloonPoppingGymEnv.agents import gnc_agent  # noqa: E402
from BalloonPoppingGymEnv.agents.gnc_agent import G  # noqa: E402

assert Path(gnc_agent.__file__).resolve().is_relative_to(ROOT)
r = np.random.default_rng(0)
e1 = e2 = 0.0
for _ in range(10000):
    p, v, b, bv, dv = (r.normal(0, s, 3) for s in (30, 15, 30, 3, 5))
    t = r.uniform(0.05, 10)
    zem = b + bv * t - (p + v * t + 0.5 * G * t**2)
    vf = v + G * t + 2 * zem / t  # N = 2 (constant acceleration) arrival velocity
    zemzev = lambda vf: 6 * zem / t**2 - 2 * (vf - (v + G * t)) / t  # noqa: E731
    a2 = 2 * zem / t**2
    e1 = max(e1, np.abs(zemzev(vf) - a2).max() / max(1.0, np.abs(a2).max()))
    e2 = max(e2, np.abs(zemzev(vf + dv) - (a2 - 2 * dv / t)).max() / max(1.0, np.abs(a2).max()))
print(f"1) max |ZEM/ZEV(v_f = v_f(N=2)) - N=2| = {e1:.2e},  max |ZEM/ZEV(v_f(N=2) + dv) - (N=2 - 2 dv / t)| = {e2:.2e}  (relative)")


class Fake(gnc_agent.GNCAgent):  # just the attributes _arrv reads
    def __init__(self, dv, vcap, dt, N=2.0):
        self.dv, self.vcap, self.a_thrust, self.stretch, self.s_mul, self.t_go_min = dv, vcap, 14.7, 1.0, 1.0, 0.3
        self.dt, self.nav_N, self.n_mul, self.coast, self.burn_left, self.target = dt, N, 1.0, 0, 9.0, 0
        self.dv_p, self.dv_v, self.dv_tgt = np.zeros(3), np.zeros(3), None


def fly(law, dv, T=3.0, dt=0.01, N=2.0, dvs=None):
    """dvs: [(time to go, dv)] = dv changed at those times (the policy's later decisions)."""
    p, v = np.zeros(3), np.array([2.0, 0.0, 12.0])
    b0, bv = np.array([15.0, -8.0, 30.0]), np.array([0.5, 0.3, 1.5])
    zem0 = b0 + bv * T - (p + v * T + 0.5 * G * T**2)
    vf0 = v + G * T + 2 * zem0 / T + dv
    f = Fake(dv, np.inf if law in ("pure", "fade") else 0.5, dt, N)
    for k in range(int(round(T / dt))):
        t = k * dt; tg = T - t  # exact time left (>= dt)
        for tc, d in dvs or ():
            if abs(tg - tc) < dt / 2:
                f.dv = d
        zem = b0 + bv * T - (p + v * tg + 0.5 * G * tg**2)
        if law == "zemzev":
            a = 6 * zem / tg**2 - 2 * (vf0 - (v + G * tg)) / tg
        else:
            a = N * zem / tg**2
            if law == "pure":  # the spec's first law: v_f = the N = 2 arrival velocity of NOW + dv (no floor / fade / cap)
                a = a - 2 * dv / tg
            elif law != "N2":
                a = a + f._arrv(tg)
        acc = a + G
        p = p + v * dt + 0.5 * acc * dt**2; v = v + acc * dt
    return np.linalg.norm(p - (b0 + bv * T)), v


for dt in (0.01, 0.001):
    print(f"2) T = 3 s, dt {dt}; miss (m), arrival velocity - N=2's (m/s); fade / cap = _arrv itself (cap inf / 0.5 a_thrust)")
    v2 = fly("N2", np.zeros(3), dt=dt)[1]
    for dv in (np.array([2.0, 0.0, 0.0]), np.array([0.0, 0.0, -3.0]), np.array([1.0, -1.0, 1.0])):
        print(f"   dv {dv}")
        for law in ("N2", "zemzev", "pure", "fade", "cap"):
            miss, vT = fly(law, dv, dt=dt)
            print(f"     {law:7s} miss {miss:7.4f}  dv_arrival {np.round(vT - v2, 3)}  (along dv / |dv|^2: {((vT - v2) @ dv) / (dv @ dv):.2f})")
print("3) held dv (1, -1, 1) from time to go T (dt 0.001): arrival velocity shift along dv / |dv|^2 (miss m)")
dv = np.array([1.0, -1.0, 1.0])
for T in (0.75, 1.0, 1.5, 2.0, 3.0, 5.0):
    v2 = fly("N2", np.zeros(3), T=T, dt=0.001)[1]
    print("   T %.2f s: " % T + "  ".join(f"{law} {((fly(law, dv, T=T, dt=0.001)[1] - v2) @ dv) / 3:.2f} (miss {fly(law, dv, T=T, dt=0.001)[0]:.3f})"
                                       for law in ("zemzev", "pure", "fade", "cap")))
print("4) _arrv (fade, cap 0.5 a_thrust), T 3 s, dt 0.001: dv changed at later decisions -> arrival shift = the last dv")
for N in (2.0, 2.3):
    v2 = fly("N2", np.zeros(3), T=3.0, dt=0.001, N=N)[1]
    for dvs in ([(2.5, np.array([0.0, 2.0, 0.0])), (2.0, np.array([0.0, 0.0, 0.0]))],
                [(2.5, np.array([-1.0, 0.0, 2.0])), (1.5, np.array([1.0, 1.0, 1.0]))]):
        miss, vT = fly("cap", np.array([1.0, -1.0, 1.0]), T=3.0, dt=0.001, N=N, dvs=dvs)
        print(f"   N {N:.1f}: dv (1,-1,1) then {[(tc, d.tolist()) for tc, d in dvs]}: miss {miss:.4f}  shift {np.round(vT - v2, 3)}")

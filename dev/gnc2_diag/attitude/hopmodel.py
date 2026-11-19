"""How many seconds per hop does finite pointing dynamics cost?  Point-mass reach-time model from each hop's start
state (the previous pop): thrust accel a (full throttle at that time), gravity, no drag, balloon = its logged path,
hit when within 1.5 m.

  T_inst : thrust direction can jump to the best constant direction instantly (the classic reach time)
  T_turn : the thrust (= nose, see common.py) starts at the actual nose direction and turns toward the best constant
           direction d with a planar rotation profile theta(t; Theta) from a closed-loop simulation of
             ours  : our cascade (k_att 3, max_rate 2.5, k_rate 10, 15 deg norm limit, 60 deg/s gimbal slew)
             fast  : the best law found by tune_fast.py (sqrt profile, 2.5 rad/s^2 decel, max rate 4, k_rate 12) on the SAME
                     hardware (alpha = K a sin g, 15 deg, 60 deg/s)
             instant rotation is T_inst
           starting from zero body rate (the actual initial rate is ignored).
  Also the rocket's own rotation at hop start is reported.
T_turn - T_inst = the pointing cost of that model; ours - fast = what a better attitude law could buy.
"""
import json
import numpy as np
from common import *
from analyze import a_full_table

HIT = 1.5
TG = np.arange(0.01, 12.0, 0.01)


def fib_sphere(n=4000):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n); th = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], -1)


DIRS = fib_sphere()


def rot_profile(Theta, law, a_thr, T=4.0, dt=0.01):
    """closed-loop planar rotation from rest toward angle Theta (rad) -> theta(t) on TG grid (held at Theta after)."""
    n = int(T / dt)
    th, w, g, g_cmd_prev = 0.0, 0.0, 0.0, 0.0
    out = np.empty(n)
    gmax, slew = 15.0, 0.6
    for i in range(n):
        e = Theta - th
        if law == "ours":
            wc = np.sign(e) * min(2.5, 3.0 * abs(e)); kr = 10.0
        else:
            wc = np.sign(e) * min(4.0, np.sqrt(2 * 2.5 * abs(e)), 6.0 * abs(e)); kr = 12.0  # best law of tune_fast.py
        gc = np.clip(kr * (wc - w), -gmax, gmax)
        # command computed from the previous state (one-step delay), applied with the slew limit
        g = g + np.clip(g_cmd_prev - g, -slew, slew)
        g_cmd_prev = gc
        al = K_ROT * a_thr * np.sin(np.radians(g))
        w += al * dt; th += w * dt
        out[i] = th
    full = np.full(len(TG), out[-1])
    full[:n] = out[:len(TG)] if n <= len(TG) else out[:len(TG)]
    return full


def slerp_integrals(prof, Theta):
    """A(t), B(t): double time-integrals of the slerp coefficients so that int int d(s) = A d0 + B d."""
    if Theta < 1e-3:
        c0, c1 = np.ones_like(prof) * 0.0, np.ones_like(prof)
    else:
        s = np.sin(Theta)
        u = np.clip(prof, -np.pi, np.pi)
        c0 = np.sin(Theta - u) / s; c1 = np.sin(u) / s
    dt = 0.01
    def dint(c):
        v = np.cumsum(c) * dt
        return np.cumsum(v) * dt
    return dint(c0), dint(c1)


_cache = {}


def profiles(law, a_thr):
    key = (law, round(a_thr * 4) / 4)
    a_thr = key[1]
    if key not in _cache:
        thetas = np.radians(np.arange(0, 179.6, 0.5))  # up to 179.5 deg (slerp is singular at 180)
        tabs = []
        for Th in thetas:
            prof = rot_profile(Th, law, a_thr)
            tabs.append(slerp_integrals(prof, Th))
        _cache[key] = (thetas, np.array([x[0] for x in tabs]), np.array([x[1] for x in tabs]))
    return _cache[key]


def reach(p0, v0, z0, a, pb, law):
    """pb (len(TG), 3) balloon path on the time grid -> (first hit time, turn angle of the best direction (deg)).
    For each tabulated turn angle Theta the best final direction d lies on the cone angle(d, z0) = Theta and is found
    exactly: displacement a (A z0 + B d) must match c = base; max over the cone of c.d = cos(Th) c.z0 + sin(Th) |c_perp|."""
    base = pb - p0 - v0 * TG[:, None] - 0.5 * G * TG[:, None] ** 2  # (T, 3) what the thrust must displace
    if law == "inst":
        ok = np.linalg.norm(base, axis=1) <= 0.5 * a * TG ** 2 + HIT
        k = np.flatnonzero(ok)
        return (TG[k[0]], np.nan) if len(k) else (np.nan, np.nan)
    thetas, A, B = profiles(law, a)  # (J,), (J, T), (J, T)
    c = base[None] - a * A[:, :, None] * z0[None, None]  # (J, T, 3)
    cz = c @ z0
    cp = np.sqrt(np.maximum((c * c).sum(-1) - cz ** 2, 0.0))
    cd = np.cos(thetas)[:, None] * cz + np.sin(thetas)[:, None] * cp
    aB = a * B
    miss2 = (c * c).sum(-1) + aB ** 2 - 2 * aB * cd
    hit = miss2 <= HIT ** 2
    anyhit = hit.any(0)
    k = np.flatnonzero(anyhit)
    if not len(k):
        return np.nan, np.nan
    return TG[k[0]], float(np.degrees(thetas[np.flatnonzero(hit[:, k[0]])[0]]))


def main():
    a_full = a_full_table()
    out = {}
    for kind in ("their", "ours_strict", "ours_free"):
        rows = []
        for n in NAMES:
            d = load(kind, n)
            t, B = d["t"], d["B"]
            for ts, te, b, first in hops(d):
                if first:
                    continue
                i = idx(t, ts)
                ii = np.clip(i + np.round(TG / DT).astype(int), 0, len(t) - 1)
                pb = B[ii, b, :3].astype(float)
                over = i + np.round(TG / DT).astype(int) >= len(t)
                if over.any():  # beyond the log: extrapolate linearly
                    vb = B[-1, b, 3:].astype(float); k0 = np.flatnonzero(over)[0]
                    pb[over] = pb[k0 - 1] + vb * (TG[over] - TG[k0 - 1])[:, None]
                a = float(a_full(t[i] - t[d["i0"]] + 1.0))
                p0, v0, z0 = d["p"][i], d["v"][i], d["zb"][i]
                ti, _ = reach(p0, v0, z0, a, pb, "inst")
                to, tho = reach(p0, v0, z0, a, pb, "ours")
                tf, thf = reach(p0, v0, z0, a, pb, "fast")
                rows.append(dict(name=n, hop=te - ts, inst=ti, ours=to, fast=tf, theta=tho, w0=float(d["wxy"][i]),
                                 vrel=float(np.linalg.norm(B[i, b, 3:] - d["v"][i])), dist=float(np.linalg.norm(B[i, b, :3] - p0))))
        out[kind] = rows
        R = {k: np.array([r[k] for r in rows], float) for k in ("hop", "inst", "ours", "fast", "theta", "w0", "dist", "vrel")}
        m = lambda x: float(np.nanmedian(x))
        mean = lambda x: float(np.nanmean(x))
        print(f"[{kind}] hops {len(rows)}  actual med {m(R['hop']):.2f} mean {mean(R['hop']):.2f} | T_inst med {m(R['inst']):.2f} mean {mean(R['inst']):.2f}"
              f" | T_turn(ours) med {m(R['ours']):.2f} mean {mean(R['ours']):.2f} | T_turn(fast) med {m(R['fast']):.2f} mean {mean(R['fast']):.2f}")
        print(f"      pointing cost ours-inst: med {m(R['ours'] - R['inst']):.3f} mean {mean(R['ours'] - R['inst']):.3f} p90 {np.nanpercentile(R['ours'] - R['inst'], 90):.3f}"
              f" | fast-inst: med {m(R['fast'] - R['inst']):.3f} mean {mean(R['fast'] - R['inst']):.3f}"
              f" | ours-fast: med {m(R['ours'] - R['fast']):.3f} mean {mean(R['ours'] - R['fast']):.3f} p90 {np.nanpercentile(R['ours'] - R['fast'], 90):.3f}")
        print(f"      actual - T_turn(ours): med {m(R['hop'] - R['ours']):.3f} mean {mean(R['hop'] - R['ours']):.3f} | "
              f"turn angle to best dir med {m(R['theta']):.1f} deg | |w| at hop start med {m(R['w0']):.2f} | dist med {m(R['dist']):.1f} m | vrel med {m(R['vrel']):.1f} m/s")
    json.dump(out, open(OUT / "hopmodel_out.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()

"""V5: independent hop reach-time model with 3-D attitude dynamics (own implementation).

For each non-first hop (start = previous pop): point mass, full-throttle thrust accel a(t) (table from our logs),
gravity, no drag, balloon = its logged path, hit radius 1.5 m.
  T_inst      : exact min time, thrust direction free (constant direction is optimal for this linear problem)
  T_grid_inst : same with the finite direction set (quantisation bias check)
  T_rest      : thrust = body z; the body starts at the ACTUAL attitude with ZERO body rate and turns toward a fixed
                direction d with our cascade (k_att 3, max_rate 2.5, k_rate 10 deg/(rad/s), |g|<=15, slew 0.6 deg/step,
                dw = K a sin g, K = 2.30); min over d (brute force: 2500 sphere dirs + 2500 around the T_inst direction)
  T_w0        : same but starting from the ACTUAL body rates at the pop
  T_fast3     : T_w0 with 3x angular acceleration (K*3) and k_att 5 / max_rate 5 / k_rate 4 (like the fast_physics run)
"""
import sys, json
import numpy as np
from multiprocessing import get_context
from base import *
from v3_hops import hoplist, a_full_tab

HIT = 1.5
KROT = 2.30
HOR = 7.0


def fib(n):
    i = np.arange(n) + 0.5
    ph = np.arccos(1 - 2 * i / n); th = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(th) * np.sin(ph), np.sin(th) * np.sin(ph), np.cos(ph)], -1)


def cap(center, radius_deg, n):
    """n roughly uniform unit vectors within radius of center."""
    u = fib(4 * n)
    c = np.cos(np.radians(radius_deg))
    # rotate the +z cap onto center
    u = u[u[:, 2] >= c][:n]
    z = center / np.linalg.norm(center)
    a = np.cross([0, 0, 1.0], z); s = np.linalg.norm(a); cth = z[2]
    if s < 1e-9:
        return u if cth > 0 else -u
    a /= s
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    Rm = np.eye(3) + s * K + (1 - cth) * K @ K
    return u @ Rm.T


def qmat(q):
    e0, e1, e2, e3 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([
        np.stack([1 - 2 * (e2 * e2 + e3 * e3), 2 * (e1 * e2 - e0 * e3), 2 * (e1 * e3 + e0 * e2)], -1),
        np.stack([2 * (e1 * e2 + e0 * e3), 1 - 2 * (e1 * e1 + e3 * e3), 2 * (e2 * e3 - e0 * e1)], -1),
        np.stack([2 * (e1 * e3 - e0 * e2), 2 * (e2 * e3 + e0 * e1), 1 - 2 * (e1 * e1 + e2 * e2)], -1)], -2)


def simulate(q0, w0, dirs, a_of_t, c_of_t, nsteps, kr=KROT, k_att=3.0, max_rate=2.5, k_rate=10.0, instant=False):
    """-> first hit time per direction (nan if none)."""
    N = len(dirs)
    q = np.tile(q0, (N, 1)); w = np.tile(np.r_[w0[:2], 0.0], (N, 1)); g = np.zeros((N, 2))
    P = np.zeros((N, 3)); V = np.zeros((N, 3))
    hit = np.full(N, np.nan)
    for k in range(nsteps):
        tk = (k + 1) * DT
        Rm = qmat(q)
        if instant:
            u = dirs
        else:
            db = np.einsum("nji,nj->ni", Rm, dirs)  # R^T d
            err = np.stack([-db[:, 1], db[:, 0], np.zeros(N)], 1)  # ez x db
            en = np.linalg.norm(err, axis=1)
            angl = np.arctan2(en, db[:, 2])
            mag = np.minimum(k_att * angl, max_rate)
            ue = err / np.maximum(en, 1e-12)[:, None]
            wc = mag[:, None] * ue
            tv = k_rate * (wc[:, :2] - w[:, :2])
            nrm = np.linalg.norm(tv, axis=1)
            tv *= np.minimum(1.0, 15.0 / np.maximum(nrm, 1e-9))[:, None]
            g = g + np.clip(tv - g, -0.6, 0.6)
            a_now = a_of_t(tk - DT)
            w[:, :2] += kr * a_now * np.sin(np.radians(g)) * DT
            # quaternion update q <- q (x) exp(w dt / 2)
            ang_ = np.linalg.norm(w, axis=1) * DT
            ax = w / np.maximum(np.linalg.norm(w, axis=1), 1e-12)[:, None]
            dq = np.c_[np.cos(ang_ / 2), np.sin(ang_ / 2)[:, None] * ax]
            q = qmul(q, dq); q /= np.linalg.norm(q, axis=1, keepdims=True)
            u = qmat(q)[:, :, 2]
        acc = a_of_t(tk - DT) * u
        V += acc * DT; P += V * DT - 0.5 * acc * DT * DT  # semi-implicit ~ exact for const acc over the step
        miss = np.linalg.norm(c_of_t(k + 1) - P, axis=1)
        new = np.isnan(hit) & (miss <= HIT)
        hit[new] = tk
        if np.isfinite(hit).any() and tk > np.nanmin(hit) + 0.02:
            break
    return hit


def work(args):
    kind, name, gx, gy = args
    af = lambda s: np.interp(s, gx, gy)
    d = get(kind, name); t = d["t"]
    rows = []
    SPH = fib(2500)
    for ts, te, b, first in hoplist(d):
        if first:
            continue
        i = ti(t, ts)
        ns = int(HOR / DT)
        tg = np.arange(ns + 1) * DT
        ii = i + np.arange(ns + 1)
        over = ii >= len(t)
        pb = d["B"][np.minimum(ii, len(t) - 1), b, :3].astype(float)
        if over.any():
            k0 = np.argmax(over); vb = d["B"][-1, b, 3:].astype(float)
            pb[over] = pb[k0 - 1] + vb * (tg[over] - tg[k0 - 1])[:, None]
        p0, v0 = d["p"][i], d["v"][i]
        C = pb - p0 - v0 * tg[:, None] - 0.5 * G * tg[:, None] ** 2
        tl = ts - d["L"]
        at = af(tl + tg)
        Dt = np.r_[0, np.cumsum(np.cumsum(at[:-1] * DT) * DT)]  # double integral of a (approx)
        # exact T_inst with the same step integrator: reach ball radius = sum over steps
        okk = np.linalg.norm(C, axis=1) <= Dt + HIT
        T_inst = tg[np.argmax(okk)] if okk.any() else np.nan
        if not np.isfinite(T_inst):
            continue
        dinst = C[np.argmax(okk)]
        z0 = d["zb"][i]
        th_needed = float(angd(z0, dinst))
        dirs = np.vstack([SPH, cap(dinst, max(40.0, th_needed + 15), 2500)])
        a_of_t = lambda s: np.interp(tl + s, gx, gy)
        c_of_t = lambda k: C[min(k, ns)]
        nst = int(min(HOR, T_inst + 3.0) / DT)
        Tg = np.nanmin(simulate(d["q"][i], d["w"][i], dirs, a_of_t, c_of_t, nst, instant=True))
        Tr = np.nanmin(simulate(d["q"][i], np.zeros(3), dirs, a_of_t, c_of_t, nst))
        Tw = np.nanmin(simulate(d["q"][i], d["w"][i], dirs, a_of_t, c_of_t, nst))
        Tf = np.nanmin(simulate(d["q"][i], d["w"][i], dirs, a_of_t, c_of_t, nst, kr=3 * KROT, k_att=5.0, max_rate=5.0, k_rate=4.0))
        rows.append(dict(kind=kind, name=name, hop=te - ts, inst=float(T_inst), ginst=float(Tg), rest=float(Tr), w0=float(Tw),
                         fast3=float(Tf), th_inst=th_needed, wxy=float(d["wxy"][i])))
    return rows


if __name__ == "__main__":
    gx, gy = a_full_tab()
    jobs = [(k, n, gx, gy) for k in ("their", "ours_strict", "ours_free") for n in NAMES]
    with get_context("spawn").Pool(4) as p:
        res = sum(p.map(work, jobs), [])
    json.dump(res, open("v5_out.json", "w"))
    for kind in ("their", "ours_strict", "ours_free"):
        R = [r for r in res if r["kind"] == kind]
        A = {k: np.array([r[k] for r in R], float) for k in ("hop", "inst", "ginst", "rest", "w0", "fast3", "th_inst", "wxy")}
        mn = lambda x: np.nanmean(x); md = lambda x: np.nanmedian(x)
        print(f"[{kind}] n={len(R)} hop mean {mn(A['hop']):.2f} | T_inst mean {mn(A['inst']):.3f} med {md(A['inst']):.2f} | grid-inst bias {mn(A['ginst']-A['inst']):.3f}"
              f" | th to T_inst dir med {md(A['th_inst']):.1f}")
        print(f"    pointing cost (vs grid-inst): rest mean {mn(A['rest']-A['ginst']):.3f} med {md(A['rest']-A['ginst']):.3f} | with actual w0 mean {mn(A['w0']-A['ginst']):.3f} med {md(A['w0']-A['ginst']):.3f}"
              f" | 3x-alpha fast w0 mean {mn(A['fast3']-A['ginst']):.3f} | w0 - fast3 mean {mn(A['w0']-A['fast3']):.3f}")
        print(f"    residual actual - T_w0 mean {mn(A['hop']-A['w0']):.3f}")

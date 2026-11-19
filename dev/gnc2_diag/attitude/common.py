"""Shared helpers for the attitude / pointing diagnosis (GNC v2).

Loads data/gnc2/*.npz and derives per-step series during the burn:
  zb  body z (nose) in world, w body rates, f specific force (dv/dt - G + drag), thrust direction / magnitude,
  inferred gimbal angle (angle between thrust and body z), hops (pop -> next pop).
"""
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data/gnc2"
OUT = Path(__file__).resolve().parent
NAMES = "10_CoolBa 10_SpaceY 11_Mochi 11_near_miss 12_Apex 13_Apex 13_Yaka 14_SpaceBerry 16_SpaceBerry 16_TeamDJ 18_Taoyuan".split()
G = np.array([0.0, 0.0, -9.81])
DT = 0.01
BURN = 30.0
# drag: k = 0.5 Cd pi r^2 / m ; accel = -k rho |v - wind| (v - wind)   (Cd 0.25, r 0.22 m); mass ~ 80 kg mid-burn
K_ROT = 2.30  # (1/m) = m * lever / I, fitted on our logged actions (validate.py, R2 0.96-0.99)
K_DRAG = 0.5 * 0.25 * np.pi * 0.22 ** 2 / 80.0


def quat_to_R(q):
    """(..., 4) scalar-first body->world -> (..., 3, 3)."""
    e0, e1, e2, e3 = [q[..., i] for i in range(4)]
    R = np.stack([
        np.stack([1 - 2 * (e2 * e2 + e3 * e3), 2 * (e1 * e2 - e0 * e3), 2 * (e1 * e3 + e0 * e2)], -1),
        np.stack([2 * (e1 * e2 + e0 * e3), 1 - 2 * (e1 * e1 + e3 * e3), 2 * (e2 * e3 - e0 * e1)], -1),
        np.stack([2 * (e1 * e3 - e0 * e2), 2 * (e2 * e3 + e0 * e1), 1 - 2 * (e1 * e1 + e2 * e2)], -1),
    ], -2)
    return R


def unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-12)


def ang(a, b):
    """angle (rad) between vectors along the last axis."""
    c = (unit(a) * unit(b)).sum(-1)
    return np.arccos(np.clip(c, -1.0, 1.0))


def load(kind, name):
    """kind: their | ours_strict | ours_free -> dict of burn-window series."""
    d = dict(np.load(DATA / f"{kind}_{name}.npz"))
    t, R = d["t"], d["R"]
    live = ~np.isnan(R[:, 0])
    launch = float(d["launch"])
    i0 = int(np.argmax(live))
    # burn window [launch, launch + BURN]
    burn = live & (t >= t[i0]) & (t <= t[i0] + BURN - 0.02)
    p, v, q, w = R[:, :3], R[:, 3:6], R[:, 6:10], R[:, 10:13]
    Rm = quat_to_R(np.where(live[:, None], q, [1, 0, 0, 0]))
    zb = Rm[..., :, 2]
    # specific force from central differences of the (exact) state velocity
    a = np.full_like(v, np.nan)
    a[1:-1] = (v[2:] - v[:-2]) / (t[2:] - t[:-2])[:, None]
    # wind from the flying balloons (they drift with the air): median horizontal velocity
    S, B = d["S"], d["B"]
    fly = (S == 1) & (B[..., 2] > 2.0 + np.nanmin(np.where(S == 1, B[..., 2], np.inf), axis=1, keepdims=True))
    wind = np.zeros((len(t), 3))
    for i in np.flatnonzero(burn):
        m = fly[i]
        if m.any():
            wind[i, :2] = np.median(B[i, m, 3:5], 0)
    vr = v - wind
    rho = 1.2
    drag = -K_DRAG * rho * np.linalg.norm(vr, axis=-1, keepdims=True) * vr
    f = a - G  # specific force (thrust + aero) / mass
    ft = f - drag
    fb = np.einsum("tji,tj->ti", Rm, f)  # specific force in body frame
    # NOTE (ActiveRocketPy flight.py u_dot): the gimbal side force enters the translational equations as the bare
    # sin(angle) (unitless, i.e. ~0.26 N), only the MOMENT uses thrust x lever -> thrust acts along body z; the gimbal
    # only rotates the body. So the gimbal is inferred from the angular acceleration: dw_xy/dt = K |T/m| sin(g)
    Tm = np.maximum(fb[:, 2], 1e-3)
    dw = np.full((len(t), 2), np.nan)
    dw[:-1] = (w[1:, :2] - w[:-1, :2]) / np.diff(t)[:, None]  # step k -> k+1 is driven by the action logged at k+1
    dw = np.r_[dw[:1] * np.nan, dw[:-1]]  # align: dw[k] = (w[k] - w[k-1]) / dt  <-> action A[k]
    sg = dw / (K_ROT * Tm[:, None])
    gxy = np.degrees(np.arcsin(np.clip(sg, -1, 1)))  # (gimbal x ~ w1, gimbal y ~ w2)
    gimb = np.hypot(gxy[:, 0], gxy[:, 1])
    d.update(t=t, live=live, burn=burn, i0=i0, launch=launch, p=p, v=v, w=w, zb=zb, Rm=Rm, f=f, ft=ft, fb=fb,
             gxy=gxy, Tm=Tm, gimb=gimb, wxy=np.linalg.norm(w[:, :2], axis=-1), wind=wind, kind=kind, name=name)
    pops = d["pops"]
    d["pops"] = pops[np.argsort(pops[:, 0])] if len(pops) else pops
    return d


def idx(t, x):
    return int(np.clip(np.searchsorted(t, x - 1e-9), 0, len(t) - 1))


def hops(d):
    """list of (t_start, t_end, balloon, first) for each pop in the burn: start = previous pop (or launch)."""
    out, prev = [], d["t"][d["i0"]]
    for tp, b in d["pops"]:
        if tp > d["t"][d["i0"]] + BURN:
            break
        out.append((prev, tp, int(b), prev == d["t"][d["i0"]]))
        prev = tp
    return out


def pct(x, qs=(50, 90, 95, 99)):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    return {f"p{q}": float(np.percentile(x, q)) for q in qs} if len(x) else {}

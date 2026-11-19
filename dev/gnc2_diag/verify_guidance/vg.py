"""Independent re-implementation for verifying the guidance_law finding (adversarial check).
Own loaders, own pop extraction from S, own thrust model (time-varying accel, double integral), own reach time."""
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
D = ROOT / "data/gnc2"
G = np.array([0.0, 0.0, -9.81])
NAMES = "10_CoolBa 10_SpaceY 11_Mochi 11_near_miss 12_Apex 13_Apex 13_Yaka 14_SpaceBerry 16_SpaceBerry 16_TeamDJ 18_Taoyuan".split()
BURN = 30.0


def load(kind, n):
    f = np.load(D / f"{kind}_{n}.npz")
    d = {k: f[k] for k in f.files}
    d["B"] = d["B"].astype(float)
    S, t = d["S"], d["t"]
    pp = []
    for b in range(S.shape[1]):
        w = np.flatnonzero(S[:, b] == 2)
        if len(w):
            pp.append((t[w[0]], b, w[0]))
    pp.sort()
    d["P"] = pp  # (t, balloon, index of first status 2)
    d["L"] = float(d["launch"])
    return d


def rotz(q):
    """body z axis in world for scalar-first body->world quaternion(s) (..., 4)"""
    e0, e1, e2, e3 = np.moveaxis(q, -1, 0)
    return np.stack([2 * (e1 * e3 + e0 * e2), 2 * (e2 * e3 - e0 * e1), 1 - 2 * (e1 ** 2 + e2 ** 2)], -1)


# thrust acceleration model: measured from their flights (upper envelope of |dv/dt - g| vs time since launch), filled
# at start of hop with the median; fallback analytic.
_AT = None


def set_athrust(tt, at):
    global _AT
    _AT = (np.asarray(tt), np.asarray(at))


def a_of(tl):
    return np.interp(tl, *_AT)


def disp_table(tl0, H=16.0, dt=0.005):
    """Displacement per unit direction under full thrust starting at time-since-launch tl0, for tau in [0, H]:
    s(tau) = int_0^tau int_0^u a(tl0 + w) dw du (a = 0 after burnout)."""
    tau = np.arange(0.0, H + dt / 2, dt)
    a = a_of(tl0 + tau) * ((tl0 + tau) < BURN)
    v = np.concatenate([[0.0], np.cumsum(0.5 * (a[1:] + a[:-1]) * dt)])
    s = np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * dt)])
    return tau, s


def reach(p, v, bp, bv, tl0, tol=1.5, H=16.0, dt=0.005):
    """Min time tau with |bp + bv tau - (p + v tau + G tau^2/2)| <= s(tau) + tol. Constant direction full thrust is the
    min-time control for free terminal velocity (costate argument), so this is the ideal point-mass lower bound (no drag)."""
    tau, s = disp_table(tl0, H, dt)
    miss = (bp - p)[None] + (bv - v)[None] * tau[:, None] - 0.5 * G[None] * tau[:, None] ** 2
    ok = np.linalg.norm(miss, axis=1) <= s + tol
    if not ok.any():
        return np.inf, None
    i = int(np.argmax(ok))
    return tau[i], miss[i] / max(np.linalg.norm(miss[i]), 1e-9)


SEED = {}
for _n in NAMES:
    SEED[_n] = int(np.load(D / f"their_{_n}.npz")["seed"])

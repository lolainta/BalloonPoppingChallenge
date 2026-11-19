"""Shared loaders / derived quantities for the launch-phase & flight-regime diagnosis (read-only on data/gnc2)."""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data/gnc2"
NAMES = "10_CoolBa 10_SpaceY 11_Mochi 11_near_miss 12_Apex 13_Apex 13_Yaka 14_SpaceBerry 16_SpaceBerry 16_TeamDJ 18_Taoyuan".split()
KINDS = ("their", "ours_strict", "ours_free")
G = np.array([0.0, 0.0, -9.81])
ELEV = 20.0
BURN = 30.0
# mass model from the given parameters (constant 1080 N thrust, liquid out 0.425 kg/s; grain mass kept -> lower bound on a)
M0 = 70 + 13.9 + 2.13 + np.pi * (0.0843**2 - 0.0295**2) * 0.2757 * 900
K_DRAG = 0.5 * 0.25 * np.pi * 0.22**2  # / m, drag accel = K rho |v| v / m


def a_max(tl):
    """Full-throttle thrust acceleration vs time since launch (liquid flow only; grain regression ignored)."""
    m = M0 - 0.425 * np.clip(tl, 0, BURN)
    return np.where((tl >= 0) & (tl <= BURN), 1080.0 / m, 0.0)


def body_z(q):
    """Body z axis (nose) in world for scalar-first quaternions q (..., 4) body->world."""
    e0, e1, e2, e3 = np.moveaxis(q, -1, 0)
    return np.stack([2 * (e1 * e3 + e0 * e2), 2 * (e2 * e3 - e0 * e1), 1 - 2 * (e1**2 + e2**2)], -1)


def incl_head(z):
    """Inclination above horizontal (deg) and heading (deg, 0 = +y north, 90 = +x east) of a direction."""
    incl = np.degrees(np.arcsin(np.clip(z[..., 2] / np.linalg.norm(z, axis=-1), -1, 1)))
    head = np.degrees(np.arctan2(z[..., 0], z[..., 1])) % 360
    return incl, head


def load(kind, name):
    d = dict(np.load(DATA / f"{kind}_{name}.npz"))
    t, R = d["t"], d["R"]
    live = np.isfinite(R[:, 0])
    i0 = int(np.argmax(live))
    d["i0"] = i0
    d["t_launch"] = float(t[i0])  # first row with a rocket state
    d["tl"] = t - t[i0]
    v = R[:, 3:6]
    dvdt = np.full_like(v, np.nan)
    dvdt[1:-1] = (v[2:] - v[:-2]) / (t[2:, None] - t[:-2, None])
    rho = 1.225 * np.clip(1.0 - 2.2558e-5 * R[:, 2:3], 0.5, 1.0) ** 4.2559
    m = (M0 - 0.425 * np.clip(d["tl"], 0, BURN))[:, None]
    drag = -K_DRAG * rho * np.linalg.norm(v, axis=1, keepdims=True) * v / m  # still air (wind ~4 m/s ignored)
    d["f"] = dvdt - G - drag  # specific force from thrust (world)
    d["alt"] = R[:, 2] - ELEV
    d["zb"] = body_z(R[:, 6:10])
    d["pops"] = np.asarray(d["pops"], float).reshape(-1, 2)
    return d

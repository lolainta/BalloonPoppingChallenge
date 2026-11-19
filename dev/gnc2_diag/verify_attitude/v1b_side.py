"""V1b: signed regression of lateral body specific force on (a) gimbal side-force term T/m sin(g), (b) aero term
(body-frame air velocity components * |v|), (c) body rates. Slope ~1 on (a) would mean the gimbal tilts the thrust."""
import numpy as np
from base import *

X, Y = [], []
for n in NAMES:
    d = get("ours_strict", n)
    b = d["burnmask"].copy(); b[:d["i0"] + 50] = False; b[-3:] = False
    f = d["f"]; v = d["v"]
    fb = np.stack([(f * d["xb"]).sum(1), (f * d["yb"]).sum(1), (f * d["zb"]).sum(1)], 1)
    vb = np.stack([(v * d["xb"]).sum(1), (v * d["yb"]).sum(1)], 1)
    sp = np.linalg.norm(v, axis=1)
    g = np.radians(d["A"][:, :2])
    # ActiveRocketPy convention: thrust1 = -sin(gy), thrust2 = sin(gx)
    gx_side = -np.sin(g[:, 1]) * fb[:, 2]; gy_side = np.sin(g[:, 0]) * fb[:, 2]
    w = d["w"]
    for ax, gs, vv, ww in ((0, gx_side, vb[:, 0], w[:, 1]), (1, gy_side, vb[:, 1], w[:, 0])):
        X.append(np.stack([gs[b], (vv * sp)[b], ww[b], np.ones(b.sum())], 1)); Y.append(fb[b, ax])
X, Y = np.concatenate(X), np.concatenate(Y)
for cols, lab in (([0, 3], "gimbal only"), ([0, 1, 2, 3], "gimbal + aero + rate")):
    c, *_ = np.linalg.lstsq(X[:, cols], Y, rcond=None)
    r2 = 1 - ((Y - X[:, cols] @ c) ** 2).mean() / Y.var()
    print(lab, "coef", np.round(c, 4), "R2", round(r2, 3))

"""V1: quaternion convention + thrust direction check; lateral body specific force vs logged gimbal."""
import numpy as np
from base import *

for kind in ("their", "ours_strict"):
    A1, A2, LAT, SIDE = [], [], [], []
    for n in NAMES:
        d = get(kind, n)
        b = d["burnmask"].copy(); b[:d["i0"] + 50] = False; b[-3:] = False
        zc = qrot(d["q"], np.array([0.0, 0.0, 1.0]), conj=True)
        A1.append(angd(d["f"][b], d["zb"][b])); A2.append(angd(d["f"][b], zc[b]))
        if kind != "their":
            fb = np.stack([(d["f"] * d["xb"]).sum(1), (d["f"] * d["yb"]).sum(1), (d["f"] * d["zb"]).sum(1)], 1)
            LAT.append(np.hypot(fb[b, 0], fb[b, 1]))
            g = np.radians(d["A"][b, :2])
            SIDE.append(fb[b, 2] * np.hypot(np.sin(g[:, 0]), np.sin(g[:, 1])))
    A1, A2 = np.concatenate(A1), np.concatenate(A2)
    print(f"[{kind}] angle(f, body z) body->world: p50 {np.median(A1):.2f} p90 {np.percentile(A1,90):.2f} p99 {np.percentile(A1,99):.2f} deg | conj convention p50 {np.median(A2):.2f}")
    if LAT:
        LAT, SIDE = np.concatenate(LAT), np.concatenate(SIDE)
        print(f"   lateral body |f| p50 {np.median(LAT):.3f} p90 {np.percentile(LAT,90):.3f}; if thrust tilted by gimbal: p50 {np.median(SIDE):.3f} p90 {np.percentile(SIDE,90):.3f}; corr {np.corrcoef(LAT, SIDE)[0,1]:.2f}")

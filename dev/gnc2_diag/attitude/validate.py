"""Validate the attitude-dynamics model on OUR runs (where the actions are logged):
 1. fit dw_xy/dt = K * (T/m) * sin(gimbal)   (K = m * lever / I)
 2. the gimbal inferred from dw/dt (common.load) vs the logged gimbal, and the saturation classification.
 3. translational check: the lateral body-frame specific force does NOT follow the gimbal (sim quirk)."""
import numpy as np
from common import *

rows, Ks = [], []
E, SAT_T, SAT_I, LAT = [], [], [], []
for n in NAMES:
    d = load("ours_strict", n)
    b = d["burn"].copy(); b[:d["i0"] + 5] = False; b[-2:] = False
    A = d["A"]
    dw = np.r_[[[np.nan, np.nan]], (d["w"][1:, :2] - d["w"][:-1, :2]) / DT]  # dw[k] <-> A[k]
    bb = b & (np.abs(A[:, :2]).max(1) > 2)
    x = np.r_[d["Tm"][bb] * np.sin(np.radians(A[bb, 0])), d["Tm"][bb] * np.sin(np.radians(A[bb, 1]))]
    y = np.r_[dw[bb, 0], dw[bb, 1]]
    K = (x @ y) / (x @ x); Ks.append(K)
    rows.append((n, K, 1 - ((y - K * x) ** 2).mean() / y.var()))
    g_true = np.hypot(A[:, 0], A[:, 1])
    E.append((d["gxy"][b] - A[b, :2]).ravel())
    SAT_T.append(g_true[b] >= 14.5); SAT_I.append(d["gimb"][b] >= 14.0)
    # lateral specific force vs gimbal side force (T/m) * sin(g)
    side = np.stack([-np.sin(np.radians(A[:, 1])), np.sin(np.radians(A[:, 0]))], -1) * d["Tm"][:, None]
    LAT.append((np.linalg.norm(d["fb"][b, :2], axis=-1), np.linalg.norm(side[b], axis=-1)))
print(f"{'run':<16}{'K (1/m)':>9}{'R2':>7}")
for r in rows:
    print(f"{r[0]:<16}{r[1]:9.3f}{r[2]:7.3f}")
E = np.concatenate(E); st, si = np.concatenate(SAT_T), np.concatenate(SAT_I)
print(f"K median {np.median(Ks):.3f}")
print(f"inferred per-axis gimbal error (deg): median |e| {np.median(np.abs(E)):.2f}, p90 {np.percentile(np.abs(E), 90):.2f}, p95 {np.percentile(np.abs(E), 95):.2f}")
print(f"saturation (|g|>=14.5 true) fraction {st.mean():.3f} ; inferred (>=14) {si.mean():.3f} ; "
      f"precision {(st & si).sum() / max(si.sum(), 1):.2f} recall {(st & si).sum() / max(st.sum(), 1):.2f}")
lat, side = [np.concatenate(z) for z in zip(*LAT)]
print(f"lateral body specific force median {np.median(lat):.2f} m/s^2 vs gimbal side force if it were T sin(g): median {np.median(side):.2f} "
      f"(p90 {np.percentile(side, 90):.2f}); corr {np.corrcoef(lat, side)[0, 1]:.2f}")

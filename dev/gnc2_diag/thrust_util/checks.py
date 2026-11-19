"""Sanity checks for util_analysis: a_max model on our full-throttle samples, spikes in |f_sp|, their f_z < 1 (dive) share."""
import sys; sys.path.insert(0, 'dev/gnc2_diag/thrust_util')
import numpy as np
from util_analysis import load, derive, NAMES
r_full, spikes = [], {}
for kind in ("their", "ours_strict", "ours_free"):
    Us, fz, sp = [], [], []
    for n in NAMES:
        d = load(kind, n); x = derive(d); m = x["burn"]
        if "A" in d:
            A = d["A"]; thr = A[:, 3]; g = np.radians(A[:, :2]); cosf = np.sqrt(1 - np.sin(g[:, 0])**2 - np.sin(g[:, 1])**2)
            k = m & (thr > 0.999); k[1:] &= thr[:-1] > 0.999; k[:-1] &= thr[1:] > 0.999
            r_full.append(x["U"][k] / cosf[k])
        Us.append(x["Usp"][m]); fz.append(x["f_thr"][m, 2]); sp.append(np.flatnonzero(m & (x["Usp"] > 1.05)).size)
    Us, fz = np.concatenate(Us), np.concatenate(fz)
    print(f"{kind:12s} |f_sp|/a_max p99 {np.percentile(Us,99):.3f} max {Us.max():.2f}  steps >1.05: {sum(sp)}  "
          f"f_z<1: {np.mean(fz<1):.4f}  f_z<0: {np.mean(fz<0):.4f}  f_z<3: {np.mean(fz<3):.4f}")
r = np.concatenate(r_full)
print(f"ours at throttle 1 (gimbal-cos corrected): U median {np.median(r):.4f} p5 {np.percentile(r,5):.4f} p95 {np.percentile(r,95):.4f} n={r.size}")

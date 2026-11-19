"""Actual miss distance at the pop (rocket centre to balloon centre, min over the pop step) and the continued
straight-line (coasting) closest approach from 0.3 s / 0.5 s before the pop."""
import numpy as np
import hops as H
for kind in ("their", "ours_strict", "ours_free"):
    dd, z3, z5 = [], [], []
    for n in H.NAMES:
        z = np.load(H.D / f"{kind}_{n}.npz"); t, R, B = z["t"], z["R"], z["B"]; launch = float(z["launch"])
        for tp, b in z["pops"]:
            if tp > launch + 30: continue
            b = int(b); i = int(np.argmin(np.abs(t - tp)))
            best = 9.0
            for j in (i - 2, i - 1, i):
                r0 = B[j, b, :3] - R[j, :3]; r1 = B[j + 1, b, :3] - R[j + 1, :3]; dr = r1 - r0
                s = np.clip(-(r0 @ dr) / max(dr @ dr, 1e-12), 0, 1); best = min(best, np.linalg.norm(r0 + s * dr))
            dd.append(best)
            for lag, L in ((30, z3), (50, z5)):
                j = i - 1 - lag
                if j < 0 or np.isnan(R[j, 0]): continue
                r, vr = B[j, b, :3] - R[j, :3], B[j, b, 3:] - R[j, 3:6]
                tc = max(-(r @ vr) / max(vr @ vr, 1e-9), 0); L.append(np.linalg.norm(r + vr * tc))
    dd = np.array(dd)
    print(f"{kind:12s} n={len(dd)} hit distance median {np.median(dd):.2f} m (p25 {np.percentile(dd,25):.2f} p75 {np.percentile(dd,75):.2f}); "
          f"frac >1.0 m {np.mean(dd > 1.0):.2f} | coast-miss from -0.3 s median {np.median(z3):.2f}, from -0.5 s {np.median(z5):.2f}")

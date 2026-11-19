"""Is the terminal miss offset (coasting closest approach from tau before the pop, rocket - balloon) on the side of the
NEXT balloon? cos between the offset and the next balloon's direction perpendicular to the approach velocity."""
import numpy as np
import hops as H
for kind in ("their", "ours_strict", "ours_free"):
    for lag in (30, 50, 80):
        cs, proj = [], []
        for n in H.NAMES:
            z = np.load(H.D / f"{kind}_{n}.npz"); t, R, B = z["t"], z["R"], z["B"]; launch = float(z["launch"])
            P = [(float(tp), int(b)) for tp, b in z["pops"] if tp <= launch + 30]
            for k in range(len(P) - 1):
                (tp, b), (_, b1) = P[k], P[k + 1]
                i = int(np.argmin(np.abs(t - tp))); j = i - 1 - lag
                if j < 0 or np.isnan(R[j, 0]) or (k > 0 and t[j] <= P[k - 1][0]): continue
                r, vr = B[j, b, :3] - R[j, :3], B[j, b, 3:] - R[j, 3:6]
                tc = max(-(r @ vr) / max(vr @ vr, 1e-9), 0)
                off = -(r + vr * tc)  # rocket - balloon at closest approach
                u = vr / np.linalg.norm(vr)
                nx = B[j, b1, :3] - B[j, b, :3]; nx = nx - (nx @ u) * u
                if np.linalg.norm(nx) < 1e-6 or np.linalg.norm(off) < 1e-6: continue
                nxh = nx / np.linalg.norm(nx)
                cs.append(off @ nxh / np.linalg.norm(off)); proj.append(off @ nxh)
        cs, proj = np.array(cs), np.array(proj)
        print(f"{kind:12s} tau=-{lag/100:.1f}s n={len(cs)} mean cos(offset, next-dir) {cs.mean():+.2f}  frac on next side {np.mean(cs > 0):.2f}  "
              f"median offset toward next {np.median(proj):+.2f} m")

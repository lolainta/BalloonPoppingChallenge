"""End of burn: chased target at burnout and its closest approach afterwards; post-burnout pops; ballistic (gravity + still-air
drag) prediction error from the burnout state; body-velocity angle at burnout."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from vc import KINDS, NAMES, get

ELEV = 20.0; KD = 0.5 * 0.25 * np.pi * 0.22**2; M_END = 70 + 2.13 + 13.9 - 0.425 * 30 + np.pi * (0.0843**2 - 0.0295**2) * 0.2757 * 900

def ballistic(p, v, T, dt=0.01):
    out = []
    for _ in range(int(round(T / dt))):
        rho = 1.225 * (1 - 2.2558e-5 * p[2]) ** 4.2559
        a = np.array([0, 0, -9.81]) - KD * rho * np.linalg.norm(v) * v / M_END
        v = v + a * dt; p = p + v * dt; out.append(p.copy())
    return np.array(out)

for k in KINDS:
    print(f"--- {k}")
    errs, angs = [], []
    for n in NAMES:
        d = get(k, n); R, B, S, t = d["R"], d["B"], d["S"], d["t"]
        ib = d["i0"] + 3000  # s = 30.00
        v = R[ib, 3:6]; ang = np.degrees(np.arccos(d["zb"][ib] @ v / np.linalg.norm(v))); angs.append(ang)
        path = ballistic(R[ib, :3].copy(), v.copy(), 3.0)
        e = [np.linalg.norm(path[int(x * 100) - 1] - R[ib + int(x * 100), :3]) if ib + int(x * 100) < len(R) and np.isfinite(R[ib + int(x * 100), 0]) else np.nan for x in (1, 2, 3)]
        errs.append(e)
        extra = ""
        if "TGT" in d:
            tg = d["TGT"][ib - 5:ib]; tg = tg[tg >= 0]
            if tg.size:
                b = int(tg[-1]); j = np.arange(ib, min(ib + 400, len(t) - 1)); j = j[np.isfinite(R[j, 0])]
                dist = np.linalg.norm(B[j, b, :3] - R[j, :3], axis=1); jm = int(np.argmin(dist))
                extra = f"target {b} status-at-end {S[-1, b]} min dist after burnout {dist[jm]:.2f} m at +{t[j[jm]] - t[ib]:.2f} s"
        post = d["ps"][d["ps"] > 30.0]
        print(f"  {n:14s} body-vel angle {ang:5.1f}  ballistic err +1/+2/+3 s {np.round(e, 1).tolist()}  post-burn pops {np.round(post - 30, 2).tolist()}  {extra}")
    E = np.array(errs)
    print(f"  == body-vel angle median {np.median(angs):.1f}; ballistic err ranges +1 s {np.nanmin(E[:,0]):.1f}-{np.nanmax(E[:,0]):.1f} (med {np.nanmedian(E[:,0]):.1f}),"
          f" +2 s {np.nanmin(E[:,1]):.1f}-{np.nanmax(E[:,1]):.1f} (med {np.nanmedian(E[:,1]):.1f}), +3 s {np.nanmin(E[:,2]):.1f}-{np.nanmax(E[:,2]):.1f} (med {np.nanmedian(E[:,2]):.1f})")

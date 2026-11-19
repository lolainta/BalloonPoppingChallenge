"""At each pop (inside the burn): angle between the nose (body z, from the logged quaternion) and the mean thrust
direction flown over the first 1.0 s of the next hop; also the thrust direction change over [pop-0.5 s, pop+1 s]."""
import sys; sys.path.insert(0, 'dev/gnc2_diag/thrust_util')
import numpy as np
from pathlib import Path
from util_analysis import load, derive, NAMES, BURN
SIMS = Path("dev/gnc2_diag/thrust_util/sims")
def _load(kind, n):
    if kind in ("their", "ours_strict", "ours_free"):
        return load(kind, n)
    return {k: v for k, v in np.load(SIMS / f"{kind}_{n}.npz").items()}
def nose(q):
    e0, e1, e2, e3 = q.T
    return np.stack([2 * (e1 * e3 + e0 * e2), 2 * (e2 * e3 - e0 * e1), 1 - 2 * (e1**2 + e2**2)], -1)
for kind in (sys.argv[1:] or ["their", "ours_strict", "ours_free"]):
    ang, ang_pre = [], []
    for n in NAMES:
        d = _load(kind, n); x = derive(d); t = x["t"]
        for tp, b in d["pops"][:-1]:
            if tp - x["L"] > BURN - 1.5:
                continue
            i = int(np.searchsorted(t, tp)); j = i + 100; k = i - 50
            if j >= len(t):
                continue
            fdir = np.nanmean(x["f_thr"][i:j], 0); fdir /= np.linalg.norm(fdir)
            nz = nose(d["R"][i:i + 1, 6:10])[0]
            fpre = np.nanmean(x["f_thr"][k:i], 0); fpre /= np.linalg.norm(fpre)
            ang.append(np.degrees(np.arccos(np.clip(nz @ fdir, -1, 1))))
            ang_pre.append(np.degrees(np.arccos(np.clip(fpre @ fdir, -1, 1))))
    ang, ang_pre = np.array(ang), np.array(ang_pre)
    print(f"{kind:12s} n={ang.size:3d} nose-at-pop vs next-1s thrust: median {np.median(ang):5.1f} deg mean {ang.mean():5.1f} | "
          f"thrust dir change (last 0.5 s before pop -> first 1 s after): median {np.median(ang_pre):5.1f} deg mean {ang_pre.mean():5.1f}")

"""Where inside a hop is time lost? For every pop-to-pop hop (ta -> tb, balloon b), every 0.05 s:
T1(t) = ideal min reach time of b from the state at t (constant-direction full thrust, coast-aware; chases.reach), and the
ideal-time loss L(t) = (t - ta) + T1(t) - T1(ta)  (0 for a perfect min-time flight; L(tb) = hop excess).
Also the pointing error: angle between the body z axis (thrust axis +- 15 deg gimbal) and the ideal min-time thrust
direction unit(miss(T1)), and the angle between the actual specific force (dv/dt - g, smoothed) and that direction.
-> loss_profile.pkl + summary by phase (first 0.5 s, 0.5-1.0 s, middle, last 0.5 s)."""
import pickle
import numpy as np, pandas as pd
from pathlib import Path
from chases import D, NAMES, reach, a_at, BURN, G, GRID
from hops import SEED

OUT = Path(__file__).resolve().parent


def ideal_dir(p, v, bp, bv, T):
    m = (bp - p) + (bv - v) * T - 0.5 * G * T ** 2
    return m / max(np.linalg.norm(m), 1e-9)


def body_z(q):
    e0, e1, e2, e3 = q
    return np.array([2 * (e1 * e3 + e0 * e2), 2 * (e2 * e3 - e0 * e1), 1 - 2 * (e1 * e1 + e2 * e2)])


rows = []
seen = set()
for n in NAMES:
    for kind in ("their", "ours_strict", "ours_free"):
        f = np.load(D / f"{kind}_{n}.npz")
        t, R, B, L, pops = f["t"], f["R"], f["B"].astype(float), float(f["launch"]), f["pops"]
        acc = np.gradient(R[:, 3:6], t, axis=0) - G
        k5 = np.ones(9) / 9  # 0.09 s smoothing of the specific force
        acc = np.stack([np.convolve(acc[:, c], k5, "same") for c in range(3)], 1)
        for k in range(1, len(pops)):
            ta, tb, b = pops[k - 1][0], pops[k][0], int(pops[k][1])
            if tb > L + BURN + 0.5:
                continue
            key = (kind, int(f["seed"]), b, round(float(ta), 2), round(float(tb), 2))
            dup = key in seen; seen.add(key)
            i0 = int(np.searchsorted(t, ta - 1e-6)); i1 = int(np.searchsorted(t, tb - 1e-6))
            T0 = None
            for i in range(i0, i1, 5):
                tl = t[i] - L
                T = reach(R[i, :3], R[i, 3:6], B[i, b, :3], B[i, b, 3:], a_at(tl), max(BURN - tl, 0.01))
                if T0 is None:
                    T0 = T
                if not np.isfinite(T):
                    continue
                u = ideal_dir(R[i, :3], R[i, 3:6], B[i, b, :3], B[i, b, 3:], T)
                z = body_z(R[i, 6:10]); fa = acc[i] / max(np.linalg.norm(acc[i]), 1e-9)
                rows.append(dict(kind=kind, name=n, dup=dup, k=k, b=b, tau=t[i] - ta, rem=tb - t[i], hop=tb - ta, T0=T0, T=T,
                                 loss=(t[i] - ta) + T - T0, ang_body=np.degrees(np.arccos(np.clip(z @ u, -1, 1))),
                                 ang_force=np.degrees(np.arccos(np.clip(fa @ u, -1, 1))), rate=np.linalg.norm(R[i, 10:13]),
                                 fmag=np.linalg.norm(acc[i])))
p = pd.DataFrame(rows)
pickle.dump(p, open(OUT / "loss_profile.pkl", "wb"))

"""Gravity-cost check: hops between consecutive pops inside the burn; dt vs rocket climb / descent / straight distance, with a
bootstrap over runs (unique flights only: duplicate strict/free flights of the same seed are collapsed) for the climb coefficient."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from vc import KINDS, NAMES, get

rng = np.random.default_rng(0)
for k in KINDS:
    runs = {}
    for n in NAMES:
        d = get(k, n)
        key = tuple(np.round(d["pops"][:, 0] - d["tL"], 2))  # identical flights collapse
        H = []
        ps = d["pops"][:, 0]
        for a, b in zip(ps[:-1], ps[1:]):
            if b - d["tL"] > 30.0: continue
            ia, ib = (int(np.argmin(np.abs(d["t"] - x))) for x in (a, b))
            dz = d["R"][ib, 2] - d["R"][ia, 2]
            H.append([b - a, max(dz, 0), max(-dz, 0), np.linalg.norm(d["R"][ib, :3] - d["R"][ia, :3])])
        runs[key] = np.array(H).reshape(-1, 4)
    R = list(runs.values()); H = np.vstack(R)
    X = np.c_[np.ones(len(H)), H[:, 1:]]
    c = np.linalg.lstsq(X, H[:, 0], rcond=None)[0]
    X2 = np.c_[np.ones(len(H)), H[:, 1]]
    c2 = np.linalg.lstsq(X2, H[:, 0], rcond=None)[0]
    bs = []
    for _ in range(2000):
        Hb = np.vstack([R[i] for i in rng.integers(0, len(R), len(R))])
        Xb = np.c_[np.ones(len(Hb)), Hb[:, 1:]]
        bs.append(np.linalg.lstsq(Xb, Hb[:, 0], rcond=None)[0][1])
    lo, hi = np.percentile(bs, [5, 95])
    print(f"{k:12s} unique flights {len(R)} hops {len(H)} median dt {np.median(H[:,0]):.2f} median climb {np.median(H[:,1]):.1f} m "
          f"median climb rate {np.median(H[:,1]/H[:,0]):.1f} m/s | fit dt = {c[0]:.2f} + {c[1]:.4f} climb + {c[2]:.4f} desc + {c[3]:.4f} dist;"
          f" climb coef 90% CI [{lo:.4f}, {hi:.4f}] | climb-only fit slope {c2[1]:.4f}")

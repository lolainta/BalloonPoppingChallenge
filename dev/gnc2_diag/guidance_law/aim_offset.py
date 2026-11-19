"""Aim offset at the hit: closest approach of the rocket's actual path (0.4 s after the pop) to where the balloon centre
would have been (constant velocity from the pop step). ~0 = aimed at the centre; ~1.5 m = grazing the edge."""
import numpy as np, pandas as pd
from chases import D, NAMES
from hops import SEED

rows, seen = [], set()
for n in NAMES:
    for kind in ("their", "ours_strict", "ours_free"):
        f = np.load(D / f"{kind}_{n}.npz")
        t, R, B, pops = f["t"], f["R"], f["B"].astype(float), f["pops"]
        for tp, b in pops:
            b = int(b); key = (kind, SEED[n], b, round(float(tp), 2))
            if key in seen:
                continue
            seen.add(key)
            i = int(np.searchsorted(t, tp - 1e-6))
            j = min(i + 40, len(t) - 1)
            if np.isnan(R[j, 0]):
                continue
            bc = B[i, b, :3][None] + B[i, b, 3:][None] * (t[i - 5:j + 1] - t[i])[:, None]
            d = np.linalg.norm(R[i - 5:j + 1, :3] - bc, axis=1)
            rows.append(dict(kind=kind, off=d.min(), vrel=np.linalg.norm(R[i, 3:6] - B[i, b, 3:])))
r = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(r.groupby("kind").off.describe(percentiles=[.1, .25, .5, .75, .9]).round(2).to_string())
print("frac aimed within 0.5 m of centre:", r.groupby("kind").off.apply(lambda x: (x < 0.5).mean()).round(2).to_dict())
print("frac grazing (> 1.0 m):", r.groupby("kind").off.apply(lambda x: (x > 1.0).mean()).round(2).to_dict())

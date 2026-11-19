"""Per-hop magnitude loss (T_mag - T_lb) split by whether our policy used stretch 1.3 during the hop."""
import sys; sys.path.insert(0, 'dev/gnc2_diag/thrust_util')
import numpy as np
from util_analysis import load, derive, hop_stats, NAMES
for kind in ("ours_strict", "ours_free"):
    rows = []
    for n in NAMES:
        d = load(kind, n); x = derive(d)
        for h in hop_stats(kind, n, d, x):
            if h["first"] or h["burn_frac"] <= 0.5 or not np.isfinite(h["T_mag"]):
                continue
            i0 = np.searchsorted(x["t"], h["t0"]); i1 = i0 + int(round(h["T_act"] / 0.01))
            st = d["STRETCH"][i0:i1]
            rows.append((np.mean(np.abs(st - 1.3) < 1e-6), h["T_mag"] - h["T_lb"], h["T_act"], h["U_hop"]))
    r = np.array(rows)
    for lab, q in (("no 1.3", r[:, 0] == 0), ("some 1.3", r[:, 0] > 0), (">50% 1.3", r[:, 0] > 0.5)):
        print(f"{kind:12s} {lab:9s} n={q.sum():3d} magL mean {r[q, 1].mean():.2f} med {np.median(r[q, 1]):.2f}  T_act mean {r[q, 2].mean():.2f}  U_hop {r[q, 3].mean():.3f}")

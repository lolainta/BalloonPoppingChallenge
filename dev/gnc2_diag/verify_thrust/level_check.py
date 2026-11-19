"""Verifier: hop magnitude loss by share of stretch 1.3 (own hop bounds), and across-hop association of U_hop with
excess hop time (T_act - T_lb)."""
import sys; sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
import json
import numpy as np
from common import *
H = json.load(open(ROOT / "dev/gnc2_diag/verify_thrust/hops.json"))
for kind in ("ours_strict", "ours_free"):
    rows = []
    for h in H[kind]:
        if h["first"] or h["burn_frac"] <= 0.5: continue
        d = load(kind, h["name"]); t = d["t"]
        i0 = int(np.searchsorted(t, h["tau0"] + float(d["launch"]) - 1e-6)); i1 = i0 + int(round(h["T_act"] / DT))
        st = d["STRETCH"][i0 + 1:i1 + 1]
        rows.append((np.mean(np.abs(st - 1.3) < 1e-6), h["T_mag"] - h["T_lb"], h["T_act"], h["U_hop"], h["T_act"] - h["T_lb"]))
    r = np.array(rows)
    for lab, q in (("no 1.3", r[:, 0] == 0), ("some 1.3", r[:, 0] > 0), (">50% 1.3", r[:, 0] > 0.5)):
        print(f"{kind:12s} {lab:9s} n={q.sum():3d} magL mean {r[q,1].mean():.3f} med {np.median(r[q,1]):.3f} T_act {r[q,2].mean():.2f} U_hop {r[q,3].mean():.3f}")
    print(f"{kind:12s} share-weighted extra magL from 1.3 over all hops: {(r[:,0]*(r[:,1]-r[r[:,0]==0,1].mean())).mean():+.3f} s")
for kind in ("their", "ours_strict", "ours_free"):
    r = np.array([(h["U_hop"], h["T_act"] - h["T_lb"], h["T_mag"] - h["T_lb"]) for h in H[kind] if not h["first"] and h["burn_frac"] > 0.5])
    c = np.polyfit(r[:, 0], r[:, 1], 1)
    print(f"{kind:12s} corr(U_hop, T_act-T_lb) {np.corrcoef(r[:,0], r[:,1])[0,1]:+.2f}  slope {c[0]:+.2f} s per unit U; corr(U_hop, magL) {np.corrcoef(r[:,0], r[:,2])[0,1]:+.2f}")

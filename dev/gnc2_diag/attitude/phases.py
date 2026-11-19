"""Where does our nose-vs-F misalignment happen, and how fast does F itself swing?
Phases of the guided burn: (a) <= 1 s after a target switch, (b) the last 0.5 s before a pop, (c) the rest.
Reports the share of time with angle > 20 deg and of the lost thrust integral |F|(1 - cos) per phase, and the angular
rate of the wanted direction F (what the attitude loop is asked to follow)."""
import numpy as np
from common import *

for kind in ("ours_strict", "ours_free"):
    tot = {p: [0.0, 0.0, 0.0] for p in "abc"}  # steps, steps>20, lost
    FR, FRE = [], []
    for n in NAMES:
        d = load(kind, n)
        t, F, tgt = d["t"], d["F"], d["TGT"]
        g = d["burn"] & np.isfinite(F).all(1); g[:d["i0"] + 50] = False
        th = np.degrees(ang(d["zb"], np.nan_to_num(F)))
        uF = unit(np.nan_to_num(F))
        fr = np.full(len(t), np.nan)
        fr[1:] = np.degrees(ang(uF[1:], uF[:-1])) / DT
        sw = np.flatnonzero((tgt[1:] != tgt[:-1]) & (tgt[1:] >= 0)) + 1
        since = np.full(len(t), np.inf)
        for s in sw:
            e = min(s + 100, len(t)); since[s:e] = np.minimum(since[s:e], t[s:e] - t[s])
        to_pop = np.full(len(t), np.inf)
        for tp, b in d["pops"]:
            k = idx(t, tp); k0 = max(k - 50, 0); to_pop[k0:k + 1] = np.minimum(to_pop[k0:k + 1], tp - t[k0:k + 1])
        ph = np.where(since <= 1.0, "a", np.where(to_pop <= 0.5, "b", "c"))
        lost = np.linalg.norm(np.nan_to_num(F), axis=1) * (1 - np.cos(np.radians(th))) * DT
        for p in "abc":
            m = g & (ph == p)
            tot[p][0] += m.sum(); tot[p][1] += (m & (th > 20)).sum(); tot[p][2] += lost[m].sum()
        ok = g & np.r_[False, g[:-1]]
        FR.append(fr[ok & (ph == "c")]); FRE.append(fr[ok & (ph == "b")])
    S = sum(v[0] for v in tot.values()); S20 = sum(v[1] for v in tot.values()); SL = sum(v[2] for v in tot.values())
    print(f"[{kind}] guided steps {int(S)}; angle>20 deg in {S20 / S:.3f} of them; lost thrust (|F|(1-cos)) {SL / 11:.1f} m/s per run")
    for p, name in zip("abc", ("<=1 s after a target switch", "last 0.5 s before a pop", "rest (mid-hop)")):
        v = tot[p]
        print(f"   {name:<28} time share {v[0] / S:.3f}  share of >20deg time {v[1] / max(S20, 1):.3f}  share of lost thrust {v[2] / SL:.3f}  (>20deg within phase {v[1] / max(v[0], 1):.3f})")
    FR, FRE = np.concatenate(FR), np.concatenate(FRE)
    print(f"   rate of the wanted direction F (deg/s): mid-hop median {np.median(FR):.1f} p90 {np.percentile(FR, 90):.1f} frac>143 (2.5 rad/s) {np.mean(FR > 143):.3f};"
          f" last 0.5 s median {np.median(FRE):.1f} p90 {np.percentile(FRE, 90):.1f} frac>143 {np.mean(FRE > 143):.3f}")

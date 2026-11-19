"""Window / phase / tail checks, recomputed independently.  Also: burnout time from the specific-force drop, decomposition
order sensitivity, and the renewal ('random phase') expectation for last-pop -> burnout."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from vc import KINDS, NAMES, get

W = {}
for k in KINDS:
    rows = []
    for n in NAMES:
        d = get(k, n); p = d["ps"]
        # burnout from the data: last step with |fsp| > 8 (thrust on) after launch
        f = np.linalg.norm(d["fsp"], axis=1); s = d["s"]
        on = np.flatnonzero((s > 1) & (f > 8))
        tb = s[on[-1]] if on.size else np.nan
        pre = p[p <= 30.0]
        gaps = np.diff(p[p <= 30.0])
        rows.append(dict(first=p[0], last=p[-1], n=len(p), tb=tb, lastpre=pre[-1] if pre.size else np.nan, post=(p > 30).sum(),
                         gaps=gaps, bins=np.histogram(p, [0, 5, 10, 30, 99])[0]))
    W[k] = rows
    F = np.array([r["first"] for r in rows]); L = np.array([r["last"] for r in rows]); N = np.array([r["n"] for r in rows])
    LP = np.array([r["lastpre"] for r in rows]); TB = np.array([r["tb"] for r in rows])
    per = np.sum(L - F) / np.sum(N - 1)
    g = np.concatenate([r["gaps"] for r in rows])
    resid = np.mean(g**2) / (2 * np.mean(g))  # renewal mean residual life
    print(f"{k:12s} first {F.mean():.2f} last {L.mean():.2f} window {np.mean(L-F):.2f} hops/run {np.mean(N-1):.2f} period {per:.2f} pops {N.sum()}"
          f" | burnout(s since first live row) min {np.nanmin(TB):.2f} max {np.nanmax(TB):.2f}"
          f" | last pre-burnout pop->30 s mean {np.mean(30-LP):.2f} med {np.median(30-LP):.2f}; P/2 {per/2:.2f}; renewal E[g^2]/2E[g] (burn gaps) {resid:.2f}; post-burn pops {sum(r['post'] for r in rows)}")
    print(f"{'':12s} bins 0-5/5-10/10-30/>30: {np.sum([r['bins'] for r in rows], 0).tolist()}  pops by t+10: {int(np.sum([r['bins'][:2] for r in rows]))}")

th, st, fr = (W[k] for k in KINDS)
def m(rows, key): return np.array([r[key] for r in rows], float)
dF = m(st, "first").mean() - m(th, "first").mean(); dL = m(th, "last").mean() - m(st, "last").mean()
Pt = np.sum(m(th, "last") - m(th, "first")) / np.sum(m(th, "n") - 1)
Ps = np.sum(m(st, "last") - m(st, "first")) / np.sum(m(st, "n") - 1)
Ws = np.mean(m(st, "last") - m(st, "first")); Wt = np.mean(m(th, "last") - m(th, "first"))
deficit = (m(th, "n").sum() - m(st, "n").sum()) / 11
print(f"\nstrict deficit {deficit:.2f} pops/run; window gap {Wt-Ws:.2f} s = first {dF:.2f} + last {dL:.2f}")
print(f"window share at THEIR period {(Wt-Ws)/Pt:.2f} pops ({(Wt-Ws)/Pt/deficit:.0%}); at OUR period {(Wt-Ws)/Ps:.2f} pops ({(Wt-Ws)/Ps/deficit:.0%})")
# last-pop part split into post-burnout extension and pre-burnout gap
LPt = np.mean(30 - m(th, "lastpre")); LPs = np.mean(30 - m(st, "lastpre"))
print(f"last-pop gap {dL:.2f} s = their post-burnout extension {m(th,'last').mean() - (30 - LPt):.2f} s + pre-burnout gap difference {LPs - LPt:.2f} s"
      f" (random-phase expectation of that difference (Ps-Pt)/2 = {(Ps-Pt)/2:.2f} s)")
# first-pop share: pops worth
print(f"first-pop gap {dF:.2f} s -> {dF/Pt:.2f} pops at their period; post-burnout pops their {sum(r['post'] for r in th)/11:.2f}/run")

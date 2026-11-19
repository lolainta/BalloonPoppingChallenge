"""Report for seq_fork.py: paired policy (free) vs best-found pops, gain distribution, exactness check (greedy route ==
free route), and per-route hop statistics (in-burn hops, as dev/gnc2_gate/gate.py stats()).
.venv/bin/python dev/gnc2_seq/s4_seq_search/report.py b4k5 [b2k3 ...]
"""
import json, sys
from collections import Counter
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "out"


def load(tag):
    return {r["seed"]: r for r in map(json.loads, open(OUT / f"{tag}.jsonl")) if "error" not in r}


def stats(route):
    P = [t for t, _ in route["pops"]]
    be = route["burn_end"] or np.inf
    nb = sum(t < be for t in P)
    return dict(n=len(P), inburn=nb, post=len(P) - nb, gaps=list(np.diff(P[:nb])),
                tmin=[x for x in route["tmin"][: max(nb - 1, 0)] if x is not None],
                bend=[x for x in route["bend"][: max(nb - 2, 0)] if x is not None])


def ms(x):
    x = np.asarray(x, float)
    return f"{x.mean():+.3f} +- {x.std(ddof=1) / np.sqrt(len(x)):.3f}"


free = {}
for r in map(json.loads, open(OUT / "free.jsonl")):
    if "error" in r:
        continue
    if r["seed"] in free:
        assert free[r["seed"]]["pops"] == r["pops"], f"free run not reproducible on seed {r['seed']}"
    free[r["seed"]] = r
tags = sys.argv[1:] or ["b4k5"]
S = {t: load(t) for t in tags}
for tag, R in S.items():
    seeds = sorted(s for s in R if s in free)
    exact = [R[s]["greedy"] is not None and R[s]["greedy"]["route"]["pops"] == free[s]["pops"] for s in seeds]
    fp = np.array([len(free[s]["pops"]) for s in seeds]); bp = np.array([R[s]["best"]["pops"] for s in seeds])
    g = bp - fp
    print(f"== {tag}: n={len(seeds)} seeds  exact greedy==free {sum(exact)}/{len(exact)}  truncated {sum(R[s]['trunc'] for s in seeds)}"
          f"  legs/seed {np.mean([R[s]['legs'] for s in seeds]):.0f}")
    print(f"policy {fp.mean():.3f}  best {bp.mean():.3f}  gain {ms(g)}  median {np.median(g):.1f}"
          f"  dist {dict(sorted(Counter(g.tolist()).items()))}  share >=1: {np.mean(g >= 1):.2f}")
    # hop stats: pooled medians over in-burn hops, and paired per-seed means
    for name, get in (("policy", lambda s: free[s]), ("best", lambda s: R[s]["best"]["route"])):
        st = [stats(get(s)) for s in seeds]
        pool = lambda k: np.concatenate([x[k] for x in st if x[k]]) if any(x[k] for x in st) else np.array([np.nan])
        print(f"  {name:6s}: pops {np.mean([x['n'] for x in st]):.2f} (in-burn {np.mean([x['inburn'] for x in st]):.2f}, "
              f"post-burn {np.mean([x['post'] for x in st]):.2f})  median gap {np.median(pool('gaps')):.2f} s  "
              f"median Tmin {np.median(pool('tmin')):.2f} s (mean {np.mean(np.minimum(pool('tmin'), 8)):.2f})  "
              f"median bend {np.median(pool('bend')):.1f} deg  first pop {np.median([get(s)['pops'][0][0] for s in seeds if get(s)['pops']]):.2f} s")
    stf = {s: stats(free[s]) for s in seeds}; stb = {s: stats(R[s]["best"]["route"]) for s in seeds}
    cl = lambda k, x: np.minimum(x, 8) if k == "tmin" else np.asarray(x)  # Tmin inf-safe cap; gaps / bends raw
    pm = lambda k, ss=None: [np.mean(cl(k, stb[s][k])) - np.mean(cl(k, stf[s][k])) for s in (ss or seeds) if stb[s][k] and stf[s][k]]
    print(f"  paired per-seed (best - policy): mean gap {ms(pm('gaps'))} s  mean Tmin {ms(pm('tmin'))} s  mean bend {ms(pm('bend'))} deg"
          f"  in-burn pops {ms([stb[s]['inburn'] - stf[s]['inburn'] for s in seeds])}  post-burn {ms([stb[s]['post'] - stf[s]['post'] for s in seeds])}")
    gs = [s for s in seeds if R[s]["best"]["pops"] > len(free[s]["pops"])]
    if gs:
        pm2 = lambda k: pm(k, gs)
        print(f"  seeds with gain (n={len(gs)}): mean Tmin {ms(pm2('tmin'))} s  mean gap {ms(pm2('gaps'))} s  mean bend {ms(pm2('bend'))} deg")
    rk = [R[s]["best"]["ranks"] for s in seeds]
    dev = [sum(1 for j in r if j > 0) for r in rk]
    print(f"  best route: forced choices/seed {np.mean([len(r) for r in rk]):.1f}, non-top-1 choices/seed {np.mean(dev):.2f}"
          f"  rank use {dict(sorted(Counter(j for r in rk for j in r).items()))}")
if len(S) > 1:
    a, b = tags[0], tags[1]
    com = sorted(set(S[a]) & set(S[b]) & set(free))
    d = [S[a][s]["best"]["pops"] - S[b][s]["best"]["pops"] for s in com]
    print(f"== {a} - {b} best pops on {len(com)} common seeds: {ms(d)}  dist {dict(sorted(Counter(d).items()))}")

"""Compare chain_sim variants: total pops, pops in the burn, hop times (duplicate flights 13_Apex / 16_SpaceBerry dropped)."""
import json, sys
import numpy as np
from pathlib import Path
here = Path(__file__).parent
DUP = {"13_Apex", "16_SpaceBerry"}
tags = sys.argv[1:]
base = None
for tag in tags:
    res = [r for r in json.load(open(here / f"chain_{tag}.json")) if r["name"] not in DUP]
    tot = sum(len(r["pops"]) for r in res)
    gaps, first, per = [], [], {}
    for r in res:
        P = [tp for tp, _ in r["pops"] if tp <= r["launch"] + 30]
        gaps += list(np.diff(P)); first += P[:1] and [P[0] - r["launch"]]
        per[r["name"]] = len(r["pops"])
    if base is None:
        base = per
    diff = [per[n] - base[n] for n in per]
    print(f"{tag:16s} pops {tot:3d} (9 flights)  per-flight diff vs {tags[0]}: {diff}  hop median {np.median(gaps):.2f} mean {np.mean(gaps):.2f} "
          f"(n={len(gaps)})  launch->1st median {np.median(first):.2f}")

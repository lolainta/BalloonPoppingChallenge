"""Cross-setting comparison: beam search is not monotone in (beam, K), so the best over all settings run on a seed is
the tightest lower bound on the sequence headroom. .venv/bin/python dev/gnc2_seq/s4_seq_search/union.py"""
import json
from collections import Counter
from pathlib import Path
import numpy as np
OUT = Path(__file__).resolve().parent / "out"
free = {r["seed"]: len(r["pops"]) for r in map(json.loads, open(OUT / "free.jsonl")) if "error" not in r}
S = {t: {r["seed"]: r["best"]["pops"] for r in map(json.loads, open(OUT / f"{t}.jsonl")) if "error" not in r}
     for t in ("b2k3", "b4k5", "b8k6")}
ms = lambda x: f"{np.mean(x):+.3f} +- {np.std(x, ddof=1) / np.sqrt(len(x)):.3f}"
seeds = sorted(set(S["b2k3"]) & set(S["b4k5"]))
u = [max(S["b2k3"][s], S["b4k5"][s]) - free[s] for s in seeds]
print(f"max(b2k3,b4k5) - policy, n={len(seeds)}: {ms(u)}  dist {dict(sorted(Counter(u).items()))}")
s8 = sorted(S["b8k6"])
for t in ("b2k3", "b4k5", "b8k6"):
    print(f"  on the {len(s8)} b8k6 seeds, {t} - policy: {ms([S[t][s] - free[s] for s in s8])}")
print(f"  max(all three) - policy on those: {ms([max(S[t][s] for t in S) - free[s] for s in s8])}")
print("  per seed (policy, b2k3, b4k5, b8k6):", [(s, free[s], S['b2k3'][s], S['b4k5'][s], S['b8k6'][s]) for s in s8])

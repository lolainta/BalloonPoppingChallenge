#!/usr/bin/env python3
# DECISION_LOG §460 G3 pooled (definition fixed in §465, before the s40l / s40m results): per seed, the mean of the two
# newest exports' pops minus the switch point's, pooled over the three branches' seeds (1,152 planned); all three needed.
# pooled < -0.15 -> revert to v2 (V4=runs/sl40/rl2_sl_s40a_v2.npz bash dev/sl4_switch.sh) and no Phase C.
#   .venv/bin/python dev/pool_g3.py
import csv, glob
import numpy as np

d = {}
for r in ("s40l", "s40m", "s40n"):
    P = {}
    for f in glob.glob(f"runs/.cmp_parts/sl4_{r}_2.3M/hour*.csv"):
        for row in csv.DictReader(open(f)):
            P.setdefault(int(row["seed"]), {})[int(row["cfg"])] = float(row["pops"])
    a = np.array([(v[1] + v[2]) / 2 - v[0] for v in P.values() if len(v) == 3])
    if len(a) > 1:
        d[r] = a
        print(f"{r}: {len(a)} seeds, {a.mean():+.3f} +- {a.std(ddof=1) / np.sqrt(len(a)):.3f}")
    else:
        print(f"{r}: no results yet")
a = np.concatenate(list(d.values()))
m, se = a.mean(), a.std(ddof=1) / np.sqrt(len(a))
verdict = "REVERT to v2, no Phase C" if m < -0.15 else "keep v4, Phase C as planned"
print(f"pooled {len(a)} seeds: {m:+.3f} +- {se:.3f} -> {verdict if len(d) == 3 else 'INCOMPLETE (' + verdict + ' so far)'}")

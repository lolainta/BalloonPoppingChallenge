"""Paired comparison of run_seeds logs on the same seeds: mean, paired diff +- SE, and mean launch time.
   .venv/bin/python dev/pair_logs.py logs/probe1_base.log logs/probe1_near2.log ...   (the first file is the baseline)"""
import re
import sys

import numpy as np


def read(path):
    out = {}
    for line in open(path):
        m = re.match(r"seed\s+(\d+)\s+popped\s+(\d+)\s+last_pop (\S+).*launch (\S+)", line)
        if m:
            out[int(m[1])] = (int(m[2]), float(m[4]))
    return out


base = read(sys.argv[1])
for path in sys.argv[1:]:
    run = read(path)
    seeds = sorted(set(base) & set(run))
    d = np.array([run[s][0] - base[s][0] for s in seeds], float)
    pops = np.array([run[s][0] for s in seeds], float)
    print(f"{path:<32} n {len(seeds):4d}  mean {pops.mean():6.3f}  diff {d.mean():+6.3f} +- {d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0:5.3f}"
          f"  launch {np.mean([run[s][1] for s in seeds]):5.1f} s")

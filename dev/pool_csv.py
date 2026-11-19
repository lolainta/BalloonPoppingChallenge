"""Pool dev/sweep_gnc.py --csv files (one per launch hour, disjoint seeds) into one paired summary in sweep_gnc's own format,
so dev/line_cmp.sh logs and the dashboard read the same (DECISION_LOG §248). The per-hour breakdown is printed in a
different format (the dashboard parses only the "[config] mean .. diff .. +- .." lines).
   python dev/pool_csv.py SCENARIO LABEL=FILE.csv [LABEL=FILE.csv ...]"""
import csv, sys

import numpy as np

sc, parts = sys.argv[1], [a.split("=", 1) for a in sys.argv[2:]]
P, names, hours = {}, {}, {}  # (label, seed) -> {cfg: pops}
for lab, f in parts:
    for r in csv.DictReader(open(f)):
        P.setdefault((lab, int(r["seed"])), {})[int(r["cfg"])] = float(r["pops"])
        names[int(r["cfg"])] = r["config"]
keys = [k for k, v in P.items() if len(v) == len(names)]  # seeds every config finished
cfgs = sorted(names)
M = np.array([[P[k][c] for c in cfgs] for k in keys])
print(f"scenario {sc} base pooled over {', '.join(l for l, _ in parts)} seeds {len(keys)}")
def two_ep_le(x, t=15):  # P(sum of two independent episodes <= t) from the empirical pops distribution (finale = two evaluations summed)
    v, n = np.unique(x, return_counts=True); p = n / n.sum()
    return sum(p[i] * p[k] for i in range(len(v)) for k in range(len(v)) if v[i] + v[k] <= t)
for j, c in enumerate(cfgs):
    d = M[:, j] - M[:, 0]
    print(f"  [{names[c] or 'base'}]  mean {M[:, j].mean():.3f}  diff {d.mean():+.3f} +- {d.std() / np.sqrt(len(d)):.3f}")
    # tail / stability line (DECISION_LOG §376; the dashboard regex skips it): p10, bad-episode shares, two-episode-sum risk
    lo6 = (M[:, j] <= 6).mean()
    print(f"      tail: p10 {np.quantile(M[:, j], 0.1):.0f}  <=6 {lo6:.1%} (d {lo6 - (M[:, 0] <= 6).mean():+.1%})  <=7 {(M[:, j] <= 7).mean():.1%}  >=10 {(M[:, j] >= 10).mean():.1%}  2ep<=15 {two_ep_le(M[:, j]):.1%}")
for lab, _ in parts:
    m = np.array([[P[k][c] for c in cfgs] for k in keys if k[0] == lab])
    if len(m):
        print(f"  ({lab}: base {m[:, 0].mean():.2f}" + "".join(f", cfg{j} {m[:, j].mean() - m[:, 0].mean():+.2f}" for j in range(1, len(cfgs))) + f"; {len(m)} seeds)")

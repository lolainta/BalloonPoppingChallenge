#!/usr/bin/env bash
# DECISION_LOG §505 pooling of the probability-ensemble screening (dev/launch_pq.sh: 2 sets pqa-b x 8 hours x 48 seeds,
# 10066200-10066967): the sets' hour CSVs merged per hour -> dev/pool_csv.py (logs/probe_pq_all.log), then the
# pre-registered rule: a config passes with diff >= +0.04 vs phe, z >= 2.24 and no hour below -0.15; a set-hour with
# < 43 of 48 seeds marks INCOMPLETE. A pass only earns a 3,072-seed confirmation (same rule), never the finale file.
#   P=pq S0=10066200 NSET=2 bash dev/pool_pq.sh
set -euo pipefail
cd ~/bpc
export P=${P:-pq} S0=${S0:-10066200} NSET=${NSET:-2}
S=abcdefgh; HRS=(0 3 6 9 12 15 18 21); miss=0
for ((k = 0; k < NSET; k++)); do for i in 0 1 2 3 4 5 6 7; do
  f=runs/.cmp_parts/probe_$P${S:$k:1}/hour${HRS[$i]}.csv; a=$((S0 + 384 * k + 48 * i))
  [ -s $f ] || { echo "MISSING $f: seeds $a-$((a + 47))"; miss=$((miss + 1)); }
done; done
[ $miss -eq 0 ] || { echo "ABORT: $miss of $((8 * NSET)) set-hour CSVs missing"; exit 1; }
D=runs/.cmp_parts/probe_${P}_all; mkdir -p $D; args=()
for h in "${HRS[@]}"; do
  f=$D/hour$h.csv; head -1 runs/.cmp_parts/probe_${P}a/hour$h.csv > $f
  for ((k = 0; k < NSET; k++)); do tail -n +2 runs/.cmp_parts/probe_$P${S:$k:1}/hour$h.csv >> $f; done; args+=(hour$h=$f)
done
nice -n 19 .venv/bin/python dev/pool_csv.py 40 "${args[@]}" > logs/probe_${P}_all.log
nice -n 19 .venv/bin/python - "${args[@]}" <<'PY' | tee -a logs/probe_${P}_all.log
import csv, os, sys
import numpy as np
S0, NSET = int(os.environ["S0"]), int(os.environ["NSET"]); NS = 384 * NSET
P, LP, H, T, names = {}, {}, {}, [0, 0, 0], {}
for a in sys.argv[1:]:
    h, f = a.split("=", 1)
    for r in csv.DictReader(open(f)):
        s, j = int(r["seed"]), int(r["cfg"])
        P.setdefault(s, {})[j] = float(r["pops"]); H[s] = h; names[j] = r["config"]; T[j] += int(r["truncated"])
        LP.setdefault(s, {})[j] = float(r["last_pop"]) if r["last_pop"] else np.nan
S = [s for s, v in P.items() if len(v) == 3]
print(f"seeds with all 3 configs {len(S)} of {NS}; wall-time truncated rows per config {T}")
a0 = lambda k, i: S0 + 384 * k + 48 * i
short = [(k, i) for k in range(NSET) for i in range(8) if sum(a0(k, i) <= s < a0(k, i) + 48 for s in S) < 43]
res = {}
for j in (1, 2):
    d = np.array([P[s][j] - P[s][0] for s in S]); m, se = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
    hrs = {h: np.mean([P[s][j] - P[s][0] for s in S if H[s] == h]) for h in sorted(set(H.values()), key=lambda x: int(x[4:]))}
    res[j] = m >= 0.04 and m / se >= 2.24 and min(hrs.values()) >= -0.15
    print(f"cfg{j} [{names[j][-40:]}]: {m:+.4f} +- {se:.4f} (z {m / se:+.2f}), hours " + " ".join(f"{v:+.2f}" for v in hrs.values())
          + f" -> {'passes' if res[j] else 'fails'} the screening")
d12 = np.array([P[s][2] - P[s][1] for s in S]); print(f"prob - logit: {d12.mean():+.4f} +- {d12.std(ddof=1) / np.sqrt(len(d12)):.4f}; "
      f"seeds where they differ: {int((d12 != 0).sum())} of {len(S)}")
print("mean last_pop (s): " + ", ".join(f"cfg{j} {np.nanmean([LP[s][j] for s in S]):.2f}" for j in (0, 1, 2)))
if short:
    print(f"INCOMPLETE: (set, hour index) with < 43 of 48 seeds: {short}")
print("VERDICT:", "screening passed by " + ", ".join(f"cfg{j}" for j in (1, 2) if res[j]) + " -> 3,072-seed confirmation (§505)"
      if any(res.values()) and not short else "no config passes the screening; the finale keeps phe")
PY

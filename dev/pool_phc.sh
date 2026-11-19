#!/usr/bin/env bash
# DECISION_LOG §460 Phase C pooling (§463 packing: 8 sets phca-h x 8 hours x 48 seeds): the sets' hour CSVs merged into 384
# seeds per hour -> dev/pool_csv.py (logs/probe_phc_all.log), then the pre-registered rule: the better config (higher mean)
# replaces cur only with diff >= +0.04, z >= 2.24 (Bonferroni over the 2 configs) and no hour below -0.15; otherwise cur.
# A missing set-hour CSV aborts with its seeds (rerun it alone: TAG=phc<set> HOURS=<h> N=48 S0=<its first seed>, the rest
# as in launch_phc.sh); a set-hour with < 43 of its 48 seeds marks the result INCOMPLETE (no switch until checked).
# §480 / §481 Phase D reuses it: P=phd S0=10054000 W1=curv5 ONLY1=1 bash dev/pool_phc.sh (sets phda-h, cfg1 = cur + v5,
# cfg2 = cur + v4w; ONLY1=1: cfg2 is a control that Phase C already rejected, so only cfg1 may replace cur, and only when
# its mean is also the higher one).
# §502 Phase F: the base is phe, not cur: P=phf S0=10062000 W1=phf W2=phf2 BN=phe BF=bpc_s40_fin_phe.py bash dev/pool_phc.sh
# (BN / BF only name the base in the VERDICT; defaults cur / bpc_s40_fb_cur.py).
# §509 probability-ensemble confirmation: P=pc S0=10067000 W1=pqp W2=pql ONLY1=solo BN=phe BF=bpc_s40_fin_phe.py bash dev/pool_phc.sh
# (ONLY1=solo: cfg1 is the only candidate, judged by its own rule whatever cfg2 does; cfg2 is a pure control).
set -euo pipefail
cd ~/bpc
export P=${P:-phc} S0=${S0:-10049000} W1=${W1:-phc} W2=${W2:-curv4} ONLY1=${ONLY1:-} BN=${BN:-cur} BF=${BF:-bpc_s40_fb_cur.py}
S=abcdefgh; HRS=(0 3 6 9 12 15 18 21); miss=0
for k in 0 1 2 3 4 5 6 7; do for i in 0 1 2 3 4 5 6 7; do
  f=runs/.cmp_parts/probe_$P${S:$k:1}/hour${HRS[$i]}.csv; a=$((S0 + 384 * k + 48 * i))
  [ -s $f ] || { echo "MISSING $f: seeds $a-$((a + 47))"; miss=$((miss + 1)); }
done; done
[ $miss -eq 0 ] || { echo "ABORT: $miss of 64 set-hour CSVs missing"; exit 1; }
D=runs/.cmp_parts/probe_${P}_all; mkdir -p $D; args=()
for h in "${HRS[@]}"; do
  f=$D/hour$h.csv; head -1 runs/.cmp_parts/probe_${P}a/hour$h.csv > $f
  for k in 0 1 2 3 4 5 6 7; do tail -n +2 runs/.cmp_parts/probe_$P${S:$k:1}/hour$h.csv >> $f; done; args+=(hour$h=$f)
done
nice -n 19 .venv/bin/python dev/pool_csv.py 40 "${args[@]}" > logs/probe_${P}_all.log
nice -n 19 .venv/bin/python - "${args[@]}" <<'PY' | tee -a logs/probe_${P}_all.log
import csv, os, sys
import numpy as np
S0, P0, W = int(os.environ["S0"]), os.environ["P"], {1: os.environ["W1"], 2: os.environ["W2"]}
BN, BF = os.environ["BN"], os.environ["BF"]
P, LP, H, T, names = {}, {}, {}, [0, 0, 0], {}
for a in sys.argv[1:]:
    h, f = a.split("=", 1)
    for r in csv.DictReader(open(f)):
        s, j = int(r["seed"]), int(r["cfg"])
        P.setdefault(s, {})[j] = float(r["pops"]); H[s] = h; names[j] = r["config"]; T[j] += int(r["truncated"])
        LP.setdefault(s, {})[j] = float(r["last_pop"]) if r["last_pop"] else np.nan  # finale tie-break: time of the last pop
S = [s for s, v in P.items() if len(v) == 3]
reg = {int(l.split(",")[0]) for l in open("data/seed_registry/used_seeds.csv") if l.split(",")[0].isdigit()}
print(f"seeds with all 3 configs {len(S)} of 3072; planned seeds registered {sum(s in reg for s in range(S0, S0 + 3072))}/3072; "
      f"wall-time truncated rows per config {T}")
a0 = lambda k, i: S0 + 384 * k + 48 * i
short = [(k, i) for k in range(8) for i in range(8) if sum(a0(k, i) <= s < a0(k, i) + 48 for s in S) < 43]
res = {}
for j in (1, 2):
    d = np.array([P[s][j] - P[s][0] for s in S]); m, se = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
    hrs = {h: np.mean([P[s][j] - P[s][0] for s in S if H[s] == h]) for h in sorted(set(H.values()), key=lambda x: int(x[4:]))}
    res[j] = (m, m >= 0.04 and m / se >= 2.24 and min(hrs.values()) >= -0.15)
    print(f"cfg{j} [{names[j]}]: {m:+.4f} +- {se:.4f} (z {m / se:+.2f}), worst hour {min(hrs, key=hrs.get)} {min(hrs.values()):+.3f} "
          f"-> {'passes' if res[j][1] else 'fails'}")
print("mean last_pop (s): " + ", ".join(f"cfg{j} {np.nanmean([LP[s][j] for s in S]):.2f}" for j in (0, 1, 2)))
for j in (1, 2):  # §509: the finale tie-break, on seeds where cfg j and the base pop the same number (>= 1); descriptive only
    e = np.array([LP[s][j] - LP[s][0] for s in S if P[s][j] == P[s][0] >= 1 and not np.isnan(LP[s][j] - LP[s][0])])
    print(f"equal pops, cfg{j} - base last_pop: {e.mean():+.2f} +- {e.std(ddof=1) / np.sqrt(len(e)):.2f} s over {len(e)} seeds")
only1 = os.environ.get("ONLY1") == "1"  # §481: cfg2 is a control (Phase C rejected it on other seeds); only cfg1 may switch
solo = os.environ.get("ONLY1") == "solo"  # §509: cfg1 is the only candidate, judged by its own rule; cfg2 a pure control
jb = 1 if solo else max((1, 2), key=lambda j: res[j][0])
if not solo and res[3 - jb][1] and not res[jb][1]:
    print(f"note: cfg{3 - jb} passes, but cfg{jb} has the higher mean and fails -> {BN} (§460: the best config must pass)")
if only1 and jb == 2:
    print(f"note: cfg2 has the higher mean but is a control (ONLY1, §481) -> {BN}")
if short:
    print(f"INCOMPLETE: (set, hour index) with < 43 of 48 seeds: {short} -> no switch until checked")
win = res[jb][1] and not short and not (only1 and jb == 2)
print("VERDICT:", f"cfg{jb} replaces {BN} -> W={W[jb]} P={P0} bash dev/pack_phc.sh" if win else f"keep {BN} ({BF})")
PY

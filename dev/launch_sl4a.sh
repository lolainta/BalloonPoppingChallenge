#!/usr/bin/env bash
# DECISION_LOG §460 B1: SL v4 swapped into the finale fallback cur as is (policy unchanged): base cur + its SL v2, cfg sl_path=V4.
# 8 hours x 96 seeds = 768 seeds 10046000-10046767 (hour index i: 10046000 + 96 i .. + 95), one 1-hour 8-core job per hour.
#   V4=runs/sl40/rl2_sl_s40lmn_v4w.npz bash dev/launch_sl4a.sh
set -euo pipefail
cd ~/bpc
V4=${V4:?set V4=<sl npz>}
n=$(cut -d, -f1 data/seed_registry/used_seeds.csv | grep -E '^[0-9]+$' | awk '$1 >= 10046000 && $1 < 10046768' | wc -l)
[ "$n" -eq 0 ] || { echo "ABORT: seed registry overlaps [10046000, 10046768)"; exit 1; }
O=runs/oracle
md5sum $O/rl3_policy_cur.npz $O/rl2_sl_cur.npz "$V4" dev/probe_cmp.sh dev/line_cmp.sbatch
AGENT_CLS=rl3 TAG=sl4a HOURS="0 3 6 9 12 15 18 21" N=96 S0=10046000 CORES=8 MEM=8G TIME=1:00:00 \
  EXCLUDE=cp11,cp14,cp18 BASE=$O/rl3_policy_cur.npz BASE_EXTRA=nav_avg=1e-4 CFGS="sl_path=$V4" \
  setsid nohup bash dev/probe_cmp.sh >> logs/probe_sched.log 2>&1 < /dev/null &
sleep 30
squeue -u $USER -h -n bpc-cmp -o "%T" | sort | uniq -c

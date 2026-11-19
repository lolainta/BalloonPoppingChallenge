#!/usr/bin/env bash
# DECISION_LOG §459: probe_sel2 repacked into 1-hour jobs. The 8 hour-jobs of launch_sel2.sh (16 cores, 3:30) got a
# Slurm start estimate of 11/6 10:02 on the full cluster (§319: only jobs of <= 1 h backfill), so they were cancelled
# before starting. Same design (§450): same soups (built 23:08 by launch_sel2.sh), same 3 configs, same 3,072 seeds
# 10042000-10045071, 384 per hour; only the packing differs: 4 sets x 8 hours x 96 seeds, 8 cores, 1 h.
# Seeds of set k (0..3), hour index i (0..7): 10042000 + 768 k + 96 i .. + 95.
set -euo pipefail
cd ~/bpc
n=$(cut -d, -f1 data/seed_registry/used_seeds.csv | grep -E '^[0-9]+$' | awk '$1 >= 10042000 && $1 < 10045072' | wc -l)
[ "$n" -eq 0 ] || { echo "ABORT: seed registry overlaps [10042000, 10045072)"; exit 1; }
O=runs/oracle
md5sum $O/rl3_policy_sel2_t2.npz $O/rl3_policy_sel2_ln.npz $O/rl3_policy_cur.npz $O/rl2_sl_sel2_t2.npz dev/probe_cmp.sh dev/line_cmp.sbatch
C1="policy_path=$O/rl3_policy_cur.npz"
C2="policy_path=$O/rl3_policy_sel2_ln.npz"
for k in 0 1 2 3; do
  tag=sel2$(echo abcd | cut -c$((k + 1)))
  AGENT_CLS=rl3 TAG=$tag HOURS="0 3 6 9 12 15 18 21" N=96 S0=$((10042000 + 768 * k)) CORES=8 MEM=8G TIME=1:00:00 \
    EXCLUDE=cp11,cp14,cp18 BASE=$O/rl3_policy_sel2_t2.npz BASE_EXTRA=nav_avg=1e-4 CFGS="$C1;$C2" \
    setsid nohup bash dev/probe_cmp.sh >> logs/probe_sched.log 2>&1 < /dev/null &
done
sleep 30
squeue -u $USER -h -n bpc-cmp -o "%i %T %N %R" | sort | uniq -c -f1 | head
squeue -u $USER -h -n bpc-cmp | wc -l

#!/usr/bin/env bash
# Seed search on this machine with one checkpoint; each 96-seed chunk is appended to a log in eval.sbatch's
# format and pushed to the cluster so dev/candidates.py picks up (and verifies) >= 9-pop results.
#   dev/local_seed_search.sh AGENT "KWARGS" SEED0 N_CHUNKS TAG
ag=$1; kw=$2; s0=$3; n=$4; tag=$5
log=~/bpc/data/eval_${tag}_0.log; mkdir -p "$(dirname $log)"
printf 'variant 0: %s\nagent %s\n' "$kw" "$ag" > $log
for c in $(seq 0 $((n - 1))); do
  lo=$((s0 + c * 96)); hi=$((lo + 95))
  AGENT=$ag .venv/bin/python dev/run_seeds.py --seeds $lo-$hi --kw $kw 2>/dev/null | grep '^seed' >> $log
  rsync -az $log user@login:~/bpc/logs/
  echo "$(date +%T) chunk $((c + 1))/$n done: $(grep -c '^seed' $log) seeds, best $(awk '/^seed/{print $4}' $log | sort -n | tail -1)"
done

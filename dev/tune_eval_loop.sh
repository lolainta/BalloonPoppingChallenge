#!/usr/bin/env bash
# Deterministic eval of every new fine-tuning checkpoint on its seed list; logs + weights go to the cluster so
# dev/candidates.py verifies >= 9-pop runs.   dev/tune_eval_loop.sh runs/tune10 "s1,s2,..."
run=$1; seeds=$2; extra=${3:-}; sfx=${4:-}; data=~/bpc/data  # extra: e.g. "sl_path=... t_min=30"; sfx: log-name suffix when several loops evaluate one run
cd "$(dirname "$0")/.."
ssh user@login "mkdir -p ~/bpc/$run"
while true; do
  for ck in $(ls $run/rl3_policy_[0-9]*.npz 2>/dev/null | sort -t_ -k3 -n); do
    step=$(basename $ck .npz | sed 's/.*_//'); log=$data/eval_tune$(basename $run)${sfx}${step}_0.log
    [ -f $log ] && continue
    printf 'variant 0: %s\nagent rl3\n' "policy_path=$ck $extra" > $log
    AGENT=rl3 .venv/bin/python dev/run_seeds.py --seeds $seeds --kw policy_path=$ck $extra 2>/dev/null | grep '^seed' >> $log
    rsync -az $ck user@login:~/bpc/$run/ && rsync -az $log user@login:~/bpc/logs/
    echo "$(date +%T) $step: $(awk '/^seed/{print $4}' $log | sort -n | uniq -c | tr '\n' ' ') | best $(awk '/^seed/{print $4"@"$2"("$6"s)"}' $log | sort -t@ -k1,1nr | head -3 | tr '\n' ' ')"
  done
  sleep 60
done

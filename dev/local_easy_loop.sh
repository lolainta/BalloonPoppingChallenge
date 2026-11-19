#!/usr/bin/env bash
# Local seed search with the best ppo8 checkpoint so far:
#  - when the cluster's deterministic eval shows a better checkpoint: fetch it and run the (growing) "easy seed"
#    list (seeds that scored >= 9 in training), skipping seeds already done for that checkpoint;
#  - otherwise, instead of idling: scan fresh seeds, 96 at a time, from RANGE0 upward (progress persisted).
# Results are pushed to the cluster's logs/ so dev/candidates.py verifies >= 9-pop runs.
#   dev/local_easy_loop.sh [RUNS="ppo8 ppo9"] [MIN_GAIN=0.0] [RANGE0=60000]   (best checkpoint across RUNS)
runs=${1:-ppo8}; gain=${2:-0.0}; range0=${3:-60000}; used=0; ck=""; data=~/bpc/data
cd "$(dirname "$0")/.."
nextf=$data/range_next.txt; [ -f $nextf ] || echo $range0 > $nextf

run_list() {  # $1 = seed list file, $2 = log
  rm -f /tmp/seedchunk_*; split -l 96 -d "$1" /tmp/seedchunk_
  for f in /tmp/seedchunk_*; do
    AGENT=rl3 .venv/bin/python dev/run_seeds.py --seeds @$f --kw $kw 2>/dev/null | grep '^seed' >> "$2"
    rsync -az "$2" user@login:~/bpc/logs/
  done
}

while true; do
  best=$(ssh -o BatchMode=yes user@login "cd ~/bpc; for r in $runs; do tail -n +2 runs/\$r/det_eval.csv 2>/dev/null | sed \"s/^/\$r,/\"; done | sort -t, -k3,3nr | head -1")
  run=${best%%,*}; best=${best#*,}
  step=${best%%,*}; mean=$(echo "$best" | cut -d, -f2)
  if [ -n "$step" ] && awk -v m="$mean" -v u="$used" -v g="$gain" 'BEGIN{exit !(m > u + g)}'; then
    ck=runs/$run/rl3_policy_$step.npz; kw="policy_path=$ck use_sl=0"
    rsync -az user@login:~/bpc/$ck runs/$run/
    ssh user@login 'cd ~/bpc; tail -n +2 -q runs/*/best_episodes.csv | sort -t, -k2,2nr | cut -d, -f4 | awk "!seen[\$0]++"' > $data/easy_seeds.txt
    log=$data/eval_easy${step}_0.log
    [ -f $log ] || printf 'variant 0: %s\nagent rl3\n' "$kw" > $log
    awk '/^seed/{print $2}' $log | sort > /tmp/done_seeds.txt
    grep -vxFf /tmp/done_seeds.txt $data/easy_seeds.txt > /tmp/todo_seeds.txt || true
    echo "$(date +%T) checkpoint $run@$step (det eval $mean): $(wc -l < /tmp/todo_seeds.txt) easy seeds to run"
    [ -s /tmp/todo_seeds.txt ] && run_list /tmp/todo_seeds.txt $log
    echo "$(date +%T) easy done: $(awk '/^seed/{print $4}' $log | sort -n | uniq -c | tr '\n' ' ') | best: $(awk '/^seed/ && $4>=9 {print $4"@"$2"("$6"s)"}' $log | sort -t@ -k1,1nr | head -3 | tr '\n' ' ')"
    used=$mean
  elif [ -n "$ck" ]; then
    lo=$(cat $nextf); seq $lo $((lo + 95)) > /tmp/range_seeds.txt; echo $((lo + 96)) > $nextf
    log=$data/eval_range${step}_0.log
    [ -f $log ] || printf 'variant 0: %s\nagent rl3\n' "$kw" > $log
    run_list /tmp/range_seeds.txt $log
    echo "$(date +%T) range $lo-$((lo + 95)) with $step: best so far $(awk '/^seed/ && $4>=9 {print $4"@"$2"("$6"s)"}' $log | sort -t@ -k1,1nr | head -2 | tr '\n' ' ')"
    continue
  fi
  sleep 300
done

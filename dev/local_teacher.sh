#!/usr/bin/env bash
# Teacher search (dev/search_teacher.py) on the local machine, for the run named in the cluster's runs/teach_run:
# newest local checkpoint of that run (dev/local_services.sh mirrors them), its launch-time branches, results pushed
# to the cluster's runs/<run>/demos_in/ where the learner picks them up.
#   setsid nohup dev/local_teacher.sh >> logs/local_teacher.log 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
R='user@login:~/bpc'  # quoted: ~ must expand on the cluster
PROCS=${PROCS:-12}; N=${N:-24}
while true; do
  run=$(ssh -o BatchMode=yes user@login 'cat ~/bpc/runs/teach_run 2>/dev/null')
  LT=$(ssh -o BatchMode=yes user@login "cat ~/bpc/runs/$run/teach_launch_times 2>/dev/null"); LT=${LT:-0}
  if [ -z "$run" ] || ! ls runs/$run/rl3_policy_[0-9]*.npz >/dev/null 2>&1; then sleep 120; continue; fi
  SEEDS=$(python3 -c "import random; r = random.Random(); print(' '.join(str(r.randint(100000, 9999999)) for _ in range($N)))")
  out=runs/$run/demos_local; mkdir -p $out
  echo "$(date +%T) $run: $N seeds, launch times $LT"
  LAUNCH_TIMES=$LT PROCS=$PROCS BEAM=4 TOPK=4 FASTK=2 SPEEDS=1 nice .venv/bin/python dev/search_teacher.py runs/$run $out $SEEDS 2>&1 \
    | grep --line-buffered "^seed"
  rsync -az --exclude='.*' $out/ $R/runs/$run/demos_in/ && echo "$(date +%T) pushed $(ls $out | wc -l) demos to $run"
done

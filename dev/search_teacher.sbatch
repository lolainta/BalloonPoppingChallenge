#!/bin/bash
# Teacher episodes by beam search (dev/search_teacher.py) on random training seeds -> runs/$RUN/demos_in/, which the
# learner of $RUN picks up for self-imitation. Low priority: fills idle cores, yields to anyone's normal jobs.
#SBATCH -p devq -J bpc-teach --time=4:00:00 --nice=1000  # QoS set by dev/keep_workers.sh
#SBATCH -o logs/teach_%j.log
cd ~/bpc
P=$(ls runs/$RUN/rl3_policy_[0-9]*.npz 2>/dev/null | sort -t_ -k3 -n | tail -1)
[ -n "$P" ] || { echo "no policy checkpoint in runs/$RUN yet"; exit 0; }
LT=${LAUNCH_TIMES:-$(cat runs/$RUN/teach_launch_times 2>/dev/null)}; LT=${LT:-0}  # launch-time branches (0 = policy's own)
NL=$(echo "$LT" | tr ',' '\n' | grep -c .)  # each branch costs a full order search: fewer seeds, same wall time
SEEDS=$(python3 -c "import random; r = random.Random($SLURM_JOB_ID); print(' '.join(str(r.randint(100000, 9999999)) for _ in range(max(1, ${NSEEDS:-200} // $NL))))")
echo "policy $P  launch times $LT"
LAUNCH_TIMES=$LT PROCS=$SLURM_CPUS_PER_TASK BEAM=${BEAM:-4} TOPK=${TOPK:-4} FASTK=${FASTK:-2} SPEEDS=${SPEEDS:-1} \
  .venv/bin/python dev/search_teacher.py "runs/$RUN" "runs/$RUN/demos_in" $SEEDS 2>&1 | grep --line-buffered "^seed"  # newest ckpt per seed

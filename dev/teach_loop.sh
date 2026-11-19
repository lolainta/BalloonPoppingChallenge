#!/usr/bin/env bash
# Keep one search-teacher job per node running for RUN (DECISION_LOG §193/§194): TEACH_NODES (default "cp09 cp11"), all of each
# node's cores, normal QoS (a devq_low teacher on cp14 was preempted after 72 min with nothing written: a seed takes ~80 min),
# beam 4 / top 5, both launch options; each job writes <run>/demos_in/<seed>.npz, the learner picks them up.
# Waits until RUN has a checkpoint. Stop: touch runs/<run>/teach_stop
# Rollout teacher (§213): TEACHER=dev/s4_rollout_teacher.py TEACH_MEM=1800 TEACH_ROUNDS=1 (ROLL_K / ROLL_M pass through)
#   setsid nohup dev/teach_loop.sh runs/s4teach >> logs/teach_loop.log 2>&1 < /dev/null &
cd ~/bpc
RUN=$1; NODES=${TEACH_NODES:-cp09 cp11}
while [ ! -e $RUN/teach_stop ]; do
  if ls $RUN/rl3_policy_[0-9]*.npz > /dev/null 2>&1 && q=$(squeue -h -u $USER -n bpc-s4teach -o "%k" 2>/dev/null); then
    for n in $NODES; do
      echo "$q" | grep -qx "teach@$(basename $RUN)@$n" && continue
      c=$(sinfo -h -n $n -o "%c"); m=$(sinfo -h -n $n -o "%m")
      k=$(( (m - 3000) / ${TEACH_MEM:-3300} )); [ $k -gt $c ] && k=$c  # MB a process: beam search 2.85 GB average, more at peaks (§204: 2800 -> OOM); rollout teacher 0.93 GB (§213: 1800)
      sbatch -q normal -w $n -c $k --mem=$((k * ${TEACH_MEM:-3300}))M --comment=teach@$(basename $RUN)@$n \
        --export=ALL,RUN=$RUN,NSEEDS=$((${TEACH_ROUNDS:-3} * k)),TEACH_BEAM=4,TEACH_TOPK=5,TEACHER=${TEACHER:-dev/s4_teacher.py} dev/s4_teacher.sbatch | sed "s/^/$(date +%T) $n ($k procs): /"
    done
  fi
  .venv/bin/python dev/patch_demos.py $RUN  # demos from tasks that started before the dt fix (§195)
  sleep 120
done

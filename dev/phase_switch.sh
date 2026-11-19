#!/usr/bin/env bash
# One-shot, in place: at AT switch RUN from "launch at 35 s or later" to free launch (user decision 2026-09-25:
# "at least 35 s at first, then let it play"): new env from runs/next_env/late5 (random launch floor 1-50 s per
# training episode, eval / submission floor 1 s) and the teacher's launch-time branches from late5.teach. Waits for a
# checkpoint, then restarts the learner (resumes it) and the workers with the new env.
#   setsid nohup dev/phase_switch.sh late2 >> logs/phase_switch.log 2>&1 < /dev/null &
cd ~/bpc
run=$1; AT=${AT:-2026-09-25 06:30}; LOGS=runs/ppo5/t1data
echo "$(date +%T) $run: free launch at $AT ($(cat runs/next_env/late5))"
while [ $(date +%s) -lt $(date -d "$AT" +%s) ]; do sleep 60; done
X=$(head -1 runs/next_env/late5 | tr -d " \n"); E="T1DATA=$LOGS,SLMODEL=runs/$run/rl2_sl.npz,RL3_ATT=1,$X"
t0=$(date +%s); until [ $(stat -c %Y runs/$run/async_last.pt) -gt $t0 ]; do sleep 5; done
echo "$X" > runs/$run/next_gen_env; echo "$E" > runs/$run/worker_env; cp runs/next_env/late5.teach runs/$run/teach_launch_times
cp runs/workers.spec runs/workers.spec.bak_phase
awk -v r="runs/$run" -v e="$E" '$1 == r {printf "%-12s %-5s %-3s %-5s %s\n", $1, $2, $3, $4, e; next} {print}' \
  runs/workers.spec > runs/workers.spec.tmp && mv runs/workers.spec.tmp runs/workers.spec
lj=$(squeue -h -u $USER -n bpc-async -o "%i %k %N" | awk -v r=$run '$2==r'); lnode=$(echo "$lj" | awk '{print $3}')
t=$(squeue -h -u $USER -n bpc-teach -o "%i %k" | awk -v c="teach@$run" '$2 == c {print $1}')
w=$(squeue -h -u $USER -n bpc-aworker -o "%i %k" | awk -v r="$run@" 'index($2, r) == 1 {print $1}')
scancel $(echo "$lj" | awk '{print $1}') $w $t 2>/dev/null
sleep 12; rm -f runs/$run/learner.addr
sbatch --comment=$run -w ${lnode:-cp18} --export=ALL,RUN=runs/$run,LOCAL=92,LAUNCH=1,BATCH=19200,$E dev/async.sbatch
echo "$(date +%T) $run switched to free launch on ${lnode:-cp18} (resumed from the checkpoint of $(date -r runs/$run/async_last.pt +%T))"

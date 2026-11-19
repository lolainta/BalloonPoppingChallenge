#!/bin/bash
# When RUN's deterministic eval stops improving (PATIENCE evals without a new best), train ONE frozen SL model
# from the latest decision logs and publish it at $RUN/rl2_sl.npz (SLMODEL of every env points there, hot-reloaded).
#SBATCH -p devq -J bpc-plateau -c 4 --mem=16G --time=40:00:00 --nice=1000
#SBATCH -o logs/plateau_%j.log
cd ~/bpc
RUN=${RUN:-runs/ppo8}; PATIENCE=${PATIENCE:-5}; LOGS=${LOGS:-runs/ppo5/t1data}
while true; do
  n=$(tail -n +2 $RUN/det_eval.csv 2>/dev/null | wc -l)
  if [ "$n" -gt "$PATIENCE" ]; then
    since=$(tail -n +2 $RUN/det_eval.csv | awk -F, -v p=$PATIENCE '{m[NR]=$2} END{best=0; for(i=1;i<=NR-p;i++) if(m[i]>best) best=m[i]; imp=0; for(i=NR-p+1;i<=NR;i++) if(m[i]>best) imp=1; print imp}')
    [ "$since" = "0" ] && break
  fi
  sleep 300
done
echo "$(date +%T) plateau: no new best in the last $PATIENCE evals"; tail -n $((PATIENCE+2)) $RUN/det_eval.csv
OMP_NUM_THREADS=4 .venv/bin/python dev/train_t1sl.py "$LOGS/*.npy" $RUN/rl2_sl_frozen.npz || exit 1
cp $RUN/rl2_sl_frozen.npz $RUN/rl2_sl.tmp.npz && mv $RUN/rl2_sl.tmp.npz $RUN/rl2_sl.npz
echo "$(date +%T) published $RUN/rl2_sl.npz: every env (learner-local and remote workers, SLMODEL points here) loads it at its next episode"

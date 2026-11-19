#!/bin/bash
# RL <-> SL loop: retrain the reach model from the PPO run's decision logs every PERIOD seconds;
# rl2_env hot-reloads SLMODEL on the next episode.
#SBATCH -p devq -J bpc-sl -c 4 --mem=16G --time=40:00:00 --nice=1000
#SBATCH -o logs/sl_loop_%j.log
cd ~/bpc
RUN=${RUN:-runs/ppo5}; PERIOD=${PERIOD:-900}
while true; do
  sleep $PERIOD
  n=$(ls $RUN/t1data/*.npy 2>/dev/null | wc -l)
  [ "$n" -lt 30 ] && { echo "$(date +%T) only $n log files, waiting"; continue; }
  OMP_NUM_THREADS=4 .venv/bin/python dev/train_t1sl.py "$RUN/t1data/*.npy" $RUN/rl2_sl_tmp.npz && \
    cp -p $RUN/rl2_sl_tmp.npz $RUN/rl2_sl_t$(date +%s).npz && mv $RUN/rl2_sl_tmp.npz $RUN/rl2_sl.npz  # stamped copies pin each version
  echo "$(date +%T) updated SL model from $n files"
done

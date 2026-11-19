#!/usr/bin/env bash
# In-place SL <-> PPO alternation (user decision 2026-09-25; replaces the generations of dev/sl_cycle.sh):
# every SL_MIN minutes train a fresh reach model from the newest decision logs and atomically replace
# runs/<run>/rl2_sl.npz. The workers hot-reload it (dev/rl2_env.py checks its mtime each episode), the learner keeps
# training (it never reads SL), and every checkpoint is saved with the SL model live at that time
# (rl2_sl_<steps>.npz), so deterministic evals stay paired. Training borrows 4 cores of the run's own learner job
# (srun --overlap), so no worker gets preempted.
# Warm start (user decision 2026-09-25): each model continues from the deployed one (its weights and input
# normalisation, SL_EPOCHS epochs at SL_LR) -> about 30% smaller output changes per swap, same or better accuracy,
# and no drop of the policy that is adapted to the previous model (from-scratch swaps cost ~0.3 pops for 20-40 min).
# FIRST_WAIT_MIN delays the first refresh after a (re)start. On the login node:
#   setsid nohup dev/sl_refresh.sh late2 >> logs/sl_refresh.log 2>&1 < /dev/null &
cd ~/bpc
run=$1; SL_MIN=${SL_MIN:-45}; LOGS=${LOGS:-runs/ppo5/t1data};  # scenario-4 runs: LOGS=runs/s4data/t1data
 SL_EPOCHS=${SL_EPOCHS:-10}; SL_LR=${SL_LR:-3e-4}
sleep $((${FIRST_WAIT_MIN:-0} * 60))
while true; do
  j=$(squeue -h -u $USER -n bpc-async -o "%i %k" | awk -v r=$run '$2 == r {print $1}' | head -1)
  if [ -n "$j" ]; then
    rm -f runs/$run/rl2_sl.next.npz
    init=""; [ -s runs/$run/rl2_sl.npz ] && init="SL_INIT=runs/$run/rl2_sl.npz SL_EPOCHS=$SL_EPOCHS SL_LR=$SL_LR"
    # SL_HID must match the deployed model for a warm start (train_t1sl refuses a silent from-scratch fallback);
    # SL_MAX_ROWS = data window, SL_MAX_DP = swap guard (see dev/train_t1sl.py)
    opt="${SL_HID:+SL_HID=$SL_HID} ${SL_MAX_ROWS:+SL_MAX_ROWS=$SL_MAX_ROWS} ${SL_MAX_DP:+SL_MAX_DP=$SL_MAX_DP}"
    srun --jobid=$j --overlap -n1 -c4 --mem=8G bash -c "OMP_NUM_THREADS=4 $init $opt .venv/bin/python dev/train_t1sl.py '$LOGS/*.npy' runs/$run/rl2_sl.next.npz" \
      < /dev/null 2>&1 | grep -E "rows|shift|shape|saved|Error" | sed "s/^/$(date +%T) /"
    # the file is written on a compute node: the login node's NFS client may still cache "no such file" from the rm
    # above (2026-09-26 13:48 a finished model was reported as failed) -> re-read the directory for up to a minute
    for i in $(seq 12); do ls runs/$run > /dev/null; [ -s runs/$run/rl2_sl.next.npz ] && break; sleep 5; done
    if [ -s runs/$run/rl2_sl.next.npz ]; then
      mkdir -p runs/$run/sl_hist; cp -p runs/$run/rl2_sl.next.npz runs/$run/sl_hist/rl2_sl_$(date +%m%d_%H%M).npz  # rollback / attribution
      mv -f runs/$run/rl2_sl.next.npz runs/$run/rl2_sl.npz && echo "$(date +%T) $run: SL replaced (${init:+warm start, $SL_EPOCHS epochs at lr $SL_LR; }workers reload it)"
    else
      echo "$(date +%T) $run: SL training failed, keeping the current model"
    fi
  else
    echo "$(date +%T) $run: no live learner, skipping"
  fi
  sleep $((SL_MIN * 60))
done

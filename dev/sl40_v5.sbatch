#!/bin/bash
# DECISION_LOG §480 (user 11/6: keep training with the new SL; iterating the SL again is fine; no freeze): SL v5 for #40 =
# v4w (runs/sl40/rl2_sl_s40lmn_v4w.npz, the live SL since 11/5 01:27) warm-started on the decision logs the v4-era policies
# wrote (t1data_s40{l,m,n}_v2 files <pid>_<time_ns>.npy, ~320 rows each). Selected by the time_ns in the file name:
#   v5w = the newest 3M rows of 11/6 01:33-07:06 (names *_17939[01]*),
#   v5b = the newest 6M rows of 11/5 20:00-11/6 07:06 (*_17938[89]* and *_17939[01]*; all after the v4 switch).
# Both 10 epochs at 3e-4 like v4w. The holdout for gate A'' (dev/sl5_fwd_eval.py, run at the end) = the files from 07:06 on
# (*_17939[2-9]*: up to 10:03, when the learners stopped, ~3 h of logs the v5s never saw).
# 14:00 11/6: the first version (job 813925) linked the files into runs/sl40/{train,fwd}_0800 and hit its 1 h limit at
# 2,921 of ~58,000 symlinks: creating links on NFS from a compute node runs at ~1/s. Now globs only (reads were fine for v4).
# 14:50 11/6: job 814001 on cp08 sat 42 min in rpc_wait: on cp08 every NFS stat takes 0.4 s and every small read 0.7 s
# (login: 1.5 ms / 7 ms), so cp08 is excluded and train_t1sl.py sorts by the file-name time instead of stat'ing.
#   sbatch dev/sl40_v5.sbatch
#SBATCH -p devq -q normal -J bpc-sl -c 8 --mem=12G --time=3:00:00 --exclude=cp08,cp11,cp14,cp18 -o logs/sl40_v5_%j.log
cd ~/bpc
PY="/tmp/llt_py/.venv/bin/python -u"; . dev/node_boot.sh
V4=runs/sl40/rl2_sl_s40lmn_v4w.npz; D='runs/s4data/t1data_s40[lmn]_v2'
echo "$(date +%T) $(hostname): training"
OMP_NUM_THREADS=4 SL_INIT=$V4 SL_HID=128,128 SL_MAX_ROWS=3000000 SL_EPOCHS=10 SL_LR=3e-4 \
  $PY dev/train_t1sl.py "$D/*_17939[01]*.npy" runs/sl40/rl2_sl_s40lmn_v5w.npz 2>&1 | sed -u "s/^/v5w: /" &
OMP_NUM_THREADS=4 SL_INIT=$V4 SL_HID=128,128 SL_MAX_ROWS=6000000 SL_EPOCHS=10 SL_LR=3e-4 \
  $PY dev/train_t1sl.py "$D/*_17938[89]*.npy|$D/*_17939[01]*.npy" runs/sl40/rl2_sl_s40lmn_v5b.npz 2>&1 | sed -u "s/^/v5b: /" &
wait
echo "$(date +%T) trained"
md5sum $V4 runs/sl40/rl2_sl_s40lmn_v5?.npz
$PY dev/sl5_fwd_eval.py
echo "$(date +%T) done"

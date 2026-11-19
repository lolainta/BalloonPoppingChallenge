#!/bin/bash
# DECISION_LOG §460 (user 11/5: "適時更新 SL 看看"): SL v4 for #40 = v2 (runs/sl40/rl2_sl_s40a_v2.npz, 10/20, the SL of every
# s40 branch since s40b) warm-started on the newest rows of the three live branches' decision logs (t1data_s40{l,m,n}_v2,
# files <pid>_<ns>.npy of ~320 rows; the glob keeps 11/3 18:00 .. 11/6, train_t1sl.py takes the newest SL_MAX_ROWS by mtime).
# Two strengths: w = 10 epochs at 3e-4 (sl_refresh's warm start), f = 30 epochs at 1e-3 (how v2 / v3 were trained).
# train_t1sl.py prints the shift the policy would see (|dp|, |dt|) and the AUC of v2 -> v4 on the newest 10 % of rows.
#   sbatch dev/sl40_v4.sbatch
#SBATCH -p devq -q normal -J bpc-sl -c 8 --mem=8G --time=0:30:00 --exclude=cp11,cp14,cp18 -o logs/sl40_v4_%j.log
cd ~/bpc
PY=/tmp/llt_py/.venv/bin/python; . dev/node_boot.sh
G='runs/s4data/t1data_s40[lmn]_v2/*_1793[78]*.npy'
V2=runs/sl40/rl2_sl_s40a_v2.npz
for v in "w 10 3e-4" "f 30 1e-3"; do
  set -- $v
  OMP_NUM_THREADS=4 SL_INIT=$V2 SL_HID=128,128 SL_MAX_ROWS=3000000 SL_EPOCHS=$2 SL_LR=$3 \
    $PY dev/train_t1sl.py "$G" runs/sl40/rl2_sl_s40lmn_v4$1.npz 2>&1 | sed "s/^/v4$1: /" &
done
wait
md5sum $V2 runs/sl40/rl2_sl_s40lmn_v4?.npz

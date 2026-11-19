#!/usr/bin/env bash
# the expanded checkpoints of B / C / F (scratch): s4res2 and s4tc grown to 8 outputs (zero), s4tc with non-zero new rows
source "$(dirname "$0")/env.sh"; cd $W
M="nice -n 10 $PY dev/arrv_check/make_arrv_npz.py"
$M $R/s4res2/rl3_policy_122313600.npz $T/r8_s4res2.npz
$M $T/rl3_policy_128851200.npz $T/r8_tc.npz
$M $T/rl3_policy_128851200.npz $T/r8_tc_dv.npz ${SCALE:-0.05}

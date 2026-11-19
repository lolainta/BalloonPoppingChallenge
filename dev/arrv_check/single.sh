#!/usr/bin/env bash
# F: single file from the non-zero 8-output npz (scratch) vs the module agent (hashes.sh C), seed 10002
source "$(dirname "$0")/env.sh"; cd $W
$PY dev/make_single_file.py $T/r8_tc_dv.npz $T/sf_r8_tc_dv.py $T/rl2_sl_128851200.npz
nice -n 10 $PY dev/arrv_check/act_hash.py $T/r8_tc_dv.npz@$T/rl2_sl_128851200.npz@10002 --module $T/sf_r8_tc_dv.py 2>&1 | grep -v Warn

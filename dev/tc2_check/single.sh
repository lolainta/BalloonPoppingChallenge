#!/usr/bin/env bash
# E: single file from the non-zero tc2 npz (scratch) vs the module agent (hashes.sh C), seed 10002
source "$(dirname "$0")/env.sh"; cd $W
$PY dev/make_single_file.py $T/tc2_1.npz $T/sf_tc2_1.py $SL8
grep -c "tc2" $T/sf_tc2_1.py
nice -n 10 $PY dev/tc2_check/act_hash.py $T/tc2_1.npz@$SL8@10002 --module $T/sf_tc2_1.py 2>&1 | grep -v Warn

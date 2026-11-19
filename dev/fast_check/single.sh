#!/usr/bin/env bash
# D: single file from the non-zero fast npz (scratch) vs the module agent (hashes.sh N), seed ${1:-10002}
source "$(dirname "$0")/env.sh"; cd $W
$PY dev/make_single_file.py $T/fast_1.npz $T/sf_fast_1.py $SLN
grep -c "fast" $T/sf_fast_1.py
nice -n 10 $PY dev/fast_check/act_hash.py $T/fast_1.npz@$SLN@${1:-10002} --module $T/sf_fast_1.py 2>&1 | grep -v Warn

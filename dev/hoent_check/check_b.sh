#!/usr/bin/env bash
source "$(dirname "$0")/env.sh"; cd $W
nice -n 15 $PY dev/hoent_check/check_b.py $POL $T/rows_1000[0-3].npz 2>&1 | grep -v Warning

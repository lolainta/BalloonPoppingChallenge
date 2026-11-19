#!/usr/bin/env bash
# C: dev/tc2_check/runc.sh NPZ [seed]
source "$(dirname "$0")/env.sh"; cd $W
PREFETCH=0 RL3_REUSE=3 nice -n 10 $PY dev/tc2_check/check_c.py "${@//\$T/$T}" 2>&1 | grep -v Warn

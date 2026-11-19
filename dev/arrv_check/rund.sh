#!/usr/bin/env bash
# D: dev/arrv_check/rund.sh NPZ [seed]   (RL3_RES3= for a 5-output checkpoint)
source "$(dirname "$0")/env.sh"; cd $W
PREFETCH=0 RL3_REUSE=3 nice -n 10 $PY dev/arrv_check/check_d.py "$@" 2>&1 | grep -v Warn

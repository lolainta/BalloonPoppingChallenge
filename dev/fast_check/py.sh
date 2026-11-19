#!/usr/bin/env bash
# dev/tc2_check/py.sh script.py args...: nice'd python with env.sh's settings ($T in args is the scratch checkpoint dir)
source "$(dirname "$0")/env.sh"; cd $W
nice -n 10 $PY "${@//\$T/$T}" 2>&1 | grep -v Warn

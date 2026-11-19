#!/usr/bin/env bash
# B (data): K episodes of the s4res2 policy on each test seed in parallel (4 cores) -> $T/rows_<seed>.npz
source "$(dirname "$0")/env.sh"; cd $W; mkdir -p $T
for s in 10000 10001 10002 10003; do
  nice -n 15 timeout ${LIMIT:-580} $PY dev/hoent_check/rollout.py $POL $s ${K:-3} $T/rows_$s.npz > $T/rows_$s.log 2>&1 &
done
wait; tail -n 4 $T/rows_*.log

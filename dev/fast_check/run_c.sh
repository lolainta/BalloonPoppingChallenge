#!/usr/bin/env bash
# C: (1) an 8-output res + tc + tc2 run from the s4next npz (RL3_FAST unset) -> async_last.pt; (2) warm start with RL3_FAST=10 from it
# (>= 3 updates, demos flowing, stale pre-change / unstamped segments sent); (3) resume again.   dev/fast_check/run_c.sh RUNDIR 1|2|3
D=$(cd "$(dirname "$0")" && pwd); RUN=$1
source $D/env.sh
case $2 in
  1) RL3_FAST= bash $D/mini_train.sh $RUN 2 --init_npz $RN; cp $RUN/async_last.pt $RUN.base.pt;;
  2) rm -f $RUN/learner.addr
     (cd $W && RL3_FAST=10 $PY $D/send_stale.py $RUN 2>&1 | grep -v Warn) &
     RL3_FAST=10 bash $D/mini_train.sh $RUN ${NUPD:-4}; wait; cp $RUN/async_last.pt $RUN.fast.pt
     grep -E "^v[0-9]" $RUN.log | grep -o "workers [0-9]* .*demos [^ ]* teacher [0-9]* dropped [^ ]*"
     cd $W && $PY $D/check_d.py $RUN.base.pt $RUN.fast.pt 2>&1 | grep -v Warn;;
  3) rm -f $RUN/learner.addr; RL3_FAST=10 bash $D/mini_train.sh $RUN ${NUPD:-2}; cd $W
     grep -E "^v[0-9]" $RUN.log | grep -o "workers [0-9]* .*demos [^ ]* teacher [0-9]* dropped [^ ]*"
     $PY $D/check_d.py $RUN.fast.pt $RUN/async_last.pt 2>&1 | grep -v Warn;;
esac

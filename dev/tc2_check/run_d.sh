#!/usr/bin/env bash
# D: (1) an 8-output res + tc run from the s4arrv npz (RL3_TCOND2 unset) -> async_last.pt; (2) warm start with RL3_TCOND2=1 from
# it (>= 3 updates, demos flowing); (3) resume again.   dev/tc2_check/run_d.sh RUNDIR 1|2|3
D=$(cd "$(dirname "$0")" && pwd); RUN=$1
source $D/env.sh
case $2 in
  1) RL3_TCOND2= bash $D/mini_train.sh $RUN 2 --init_npz $R8; cp $RUN/async_last.pt $RUN.base.pt;;
  2) rm -f $RUN/learner.addr
     RL3_TCOND2=1 bash $D/mini_train.sh $RUN ${NUPD:-4}; cp $RUN/async_last.pt $RUN.tc2.pt
     grep -E "^v[0-9]" $RUN.log | grep -o "workers [0-9]* .*demos [^ ]* teacher [0-9]* dropped [^ ]*"
     cd $W && $PY $D/check_d.py $RUN.base.pt $RUN.tc2.pt 2>&1 | grep -v Warn;;
  3) rm -f $RUN/learner.addr; RL3_TCOND2=1 bash $D/mini_train.sh $RUN 2; cd $W
     grep -E "^v[0-9]" $RUN.log | grep -o "workers [0-9]* .*demos [^ ]* teacher [0-9]* dropped [^ ]*"
     $PY $D/check_d.py $RUN.tc2.pt $RUN/async_last.pt 2>&1 | grep -v Warn;;
esac

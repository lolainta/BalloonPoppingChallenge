#!/usr/bin/env bash
# D: (1) a short 5-output run without tc from the s4res2 npz -> async_last.pt; (2) warm start with RL3_TCOND=1 from it
# (>= 3 updates, demos flowing); (3) resume again from the tc checkpoint.   dev/tcond_check/run_d.sh RUNDIR STAGE(1|2|3)
D=$(cd "$(dirname "$0")" && pwd); RUN=$1
export RL3_WAIT_NODISCOUNT=1 SIL_MIN=${SIL_MIN:-1} BATCH=${BATCH:-128}
case $2 in
  1) RL3_TCOND= bash $D/mini_train.sh $RUN 2 --init_npz ${TC_TMP:-$TMPDIR/ck}/rl3_policy_116668800.npz
     cp $RUN/async_last.pt $RUN.base.pt;;
  2) RL3_TCOND=1 bash $D/mini_train.sh $RUN 4; cp $RUN/async_last.pt $RUN.tc.pt
     source $D/env.sh; $PY $D/check_d.py $RUN.base.pt $RUN.tc.pt;;
  3) RL3_TCOND=1 bash $D/mini_train.sh $RUN 2; source $D/env.sh; $PY $D/check_d.py $RUN.tc.pt $RUN/async_last.pt;;
esac

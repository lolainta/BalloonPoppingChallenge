#!/usr/bin/env bash
# E: (1) a 5-output + tc run from the s4tc npz (RL3_RES3 unset) -> async_last.pt; (2) warm start with RL3_RES3=1 from it
# (>= 3 updates, demos flowing, a stale 10-column segment / demo sent); (3) resume again.   dev/arrv_check/run_e.sh RUNDIR 1|2|3
D=$(cd "$(dirname "$0")" && pwd); RUN=$1
source $D/env.sh
case $2 in
  1) RL3_RES3= bash $D/mini_train.sh $RUN 2 --init_npz $T/rl3_policy_128851200.npz; cp $RUN/async_last.pt $RUN.base.pt;;
  2) rm -f $RUN/learner.addr $RUN/live_policy.npz
     (cd $W && ARRV_RUN=$RUN.stale timeout 600 nice -n 10 $PY $D/send_stale.py $RUN 2>&1 | grep -v Warn) &
     RL3_RES3=1 bash $D/mini_train.sh $RUN ${NUPD:-4}; wait; cp $RUN/async_last.pt $RUN.r8.pt
     grep -E "^v[0-9]" $RUN.log | grep -o "demos [^ ]* teacher [0-9]* dropped [^ ]*"
     cd $W && $PY $D/check_e.py $RUN.base.pt ${RUN}_v0.npz $RUN.r8.pt 2>&1 | grep -v Warn;;
  3) rm -f $RUN/learner.addr; RL3_RES3=1 bash $D/mini_train.sh $RUN 2; cd $W
     $PY -c "import torch; [print(f, {k: v for k, v in torch.load(f, map_location='cpu').items() if k in ('steps', 'version')}) for f in ('$RUN.r8.pt', '$RUN/async_last.pt')]";;
esac

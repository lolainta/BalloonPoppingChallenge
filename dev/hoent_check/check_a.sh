#!/usr/bin/env bash
# A: NUPD deterministic learner updates (deployed env, RL3_NEWENT=0.35, SIL demos, init from the s4res2 npz) with the
# pre-change async_ppo.py ($T/pre, = git show greenbean-dev:dev/async_ppo.py) and with this tree's, RL3_HOENT unset;
# then this tree's with RL3_HOENT=0.3 (must differ: the comparison can fail).
source "$(dirname "$0")/env.sh"; cd $W
export RL3_NEWENT=0.35; unset RL3_HOENT
ROWS=$(ls $T/rows_1000[0-3].npz | paste -sd,); N=${NUPD:-2}
for v in pre new hoent; do
  rm -rf $T/a_$v; S=$W/dev/async_ppo.py; [ $v = pre ] && S=$T/pre/async_ppo.py
  ( [ $v = hoent ] && export RL3_HOENT=0.3; nice -n 15 timeout 590 $GPY dev/hoent_check/det_learn.py $S $T/a_$v $N $T/a_$v.pt $ROWS \
    --init_npz $POL --batch 256 2>&1 | grep -v -i warn | grep -v "^\s*$" | sed "s/^/[$v] /" )
done
$GPY dev/hoent_check/cmp_ck.py $T/a_pre.pt $T/a_new.pt
echo "--- sanity: new with RL3_HOENT=0.3 vs pre"
$GPY dev/hoent_check/cmp_ck.py $T/a_pre.pt $T/a_hoent.pt

#!/usr/bin/env bash
# D: NUPD deterministic learner updates (deployed env, RL3_NEWENT=0.35, init from the s4res2 npz) with RL3_HOENT=$HO and
# without; then the HOENT run restarted from its async_last.pt for 2 more updates (log_alpha_h restored); then the loss
# quantity of the exported policies on all rows (check_d.py).
source "$(dirname "$0")/env.sh"; cd $W
export RL3_NEWENT=0.35; unset RL3_HOENT
ROWS=$(ls $T/rows_1000[0-3].npz | paste -sd,); N=${NUPD:-8}; HO=${HO:-0.3}
for v in off on; do
  rm -rf $T/d_$v
  ( [ $v = on ] && export RL3_HOENT=$HO; nice -n 15 timeout 590 $GPY dev/hoent_check/det_learn.py dev/async_ppo.py $T/d_$v $N $T/d_$v.pt $ROWS \
    --init_npz $POL --batch 256 2>&1 | grep -E "^v[0-9]|resumed|initialised|det_learn" | sed "s/^/[$v] /" )
done
echo "--- restart of the HOENT run"
( export RL3_HOENT=$HO; nice -n 15 timeout 590 $GPY dev/hoent_check/det_learn.py dev/async_ppo.py $T/d_on 2 $T/d_on2.pt $ROWS \
  --init_npz $POL --batch 256 2>&1 | grep -E "^v[0-9]|resumed|initialised|det_learn" | sed "s/^/[on restart] /" )
$GPY -c "import torch; [print(f, {k: v for k, v in torch.load(f, map_location='cpu').items() if k.startswith(('log_alpha', 'steps', 'version'))}) for f in ('$T/d_on.pt', '$T/d_on2.pt')]"
echo "status.json: $(cat $T/d_on/status.json)"
nice -n 15 $PY dev/hoent_check/check_d.py $ROWS $POL $(ls -v $T/d_off/rl3_policy_*.npz | tail -1) $(ls -v $T/d_on/rl3_policy_*.npz) 2>&1 | grep -v Warning

#!/usr/bin/env bash
# E (extra): a 5-output run WITHOUT tc (s4res2 npz) -> warm start with RL3_RES3=1 RL3_TCOND=1 (res expanded + tc added).
#   dev/arrv_check/run_e_notc.sh RUNDIR
D=$(cd "$(dirname "$0")" && pwd); RUN=$1
source $D/env.sh
RL3_RES3= RL3_TCOND= bash $D/mini_train.sh $RUN 2 --init_npz $R/s4res2/rl3_policy_122313600.npz; cp $RUN/async_last.pt $RUN.base.pt
rm -f $RUN/learner.addr
RL3_RES3=1 RL3_TCOND=1 bash $D/mini_train.sh $RUN 2
cd $W && $PY -c "
import torch
B, A = (torch.load(f, map_location='cpu') for f in ('$RUN.base.pt', '$RUN/async_last.pt'))
print('base', {k: tuple(v.shape) for k, v in B['net'].items() if k.startswith(('res', 'tc.'))}, B['steps'])
print('after', {k: tuple(v.shape) for k, v in A['net'].items() if k.startswith(('res', 'tc.'))}, A['steps'])
print('old res rows moved by <=', float((A['net']['res.weight'][:5] - B['net']['res.weight']).abs().max()), ' groups', [len(g['params']) for g in A['opt']['param_groups']],
      ' group 0 steps base / after', max(float(s['step']) for i, s in B['opt']['state'].items() if i in B['opt']['param_groups'][0]['params']),
      max(float(s['step']) for i, s in A['opt']['state'].items() if i in A['opt']['param_groups'][0]['params']))"

#!/usr/bin/env bash
# D: dev/res_check/mini_train.sh with LOCAL=3 local workers (shared machine: learner + 3 + a 2-process sweep <= 6 cores), the code of THIS tree
# (PYTHONPATH), the SL / init checkpoint by absolute path, processes tagged EVALOPTS_RUN=<run> for the cleanup.
#   dev/evalopts_check/mini_train.sh RUN NUPD [learner args...]   (RL3_RES etc. from the environment)
cd "$(dirname "$0")/../.."
RUN=$1; NUPD=$2; shift 2
export RL3_SCEN=4 RL3_ATTC=1 RL3_T_MIN=1 RL3_FREE_LAUNCH=1 RL3_KEEP_LAUNCH=1 RL3_FASTWIND=1 RL3_NEWENT=0.35 \
  GNC_KW="k_att:3+max_rate:2.5+coast:1+min_alt:3+lag_comp:1+bwind:4+margin:0+eob:2+eob_tau:3+eob_T:4" \
  SLMODEL=${SLMODEL:-./dev/res_check/tmp/rl2_sl_86630400.npz} EVALOPTS_RUN=$RUN \
  PYTHONPATH=$PWD
nice -n 10 ./.venv-gpu/bin/python dev/async_ppo.py learn --run $RUN --launch --local ${LOCAL:-3} \
  --batch 400 --ckpt_every 0 "$@" > $RUN.log 2>&1 &
L=$!
t0=$(date +%s)
while kill -0 $L 2>/dev/null && [ $(grep -c '^v[0-9]' $RUN.log) -lt $NUPD ] && [ $(( $(date +%s) - t0 )) -lt ${LIMIT:-560} ]; do sleep 5; done
kill -INT $L 2>/dev/null; sleep 3; kill $L 2>/dev/null
for sig in TERM KILL; do
  for e in /proc/[0-9]*/environ; do tr '\0' '\n' 2>/dev/null < $e | grep -qx "EVALOPTS_RUN=$RUN" && kill -$sig $(basename $(dirname $e)) 2>/dev/null; done
  sleep 2
done
wait $L 2>/dev/null
echo "elapsed $(( $(date +%s) - t0 )) s"; grep -v "^\s*$" $RUN.log | grep -v Warning | tail -12

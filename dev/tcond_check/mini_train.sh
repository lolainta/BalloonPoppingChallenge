#!/usr/bin/env bash
# D: a local learner (+ LOCAL workers, default 4) on scenario 4 until NUPD PPO updates are done (or LIMIT s), then stopped.
#   dev/tcond_check/mini_train.sh RUN NUPD [learner args...]   (RL3_TCOND etc. from the environment; env.sh sets the rest)
TC=${RL3_TCOND-1}
source "$(dirname "$0")/env.sh"; cd $W
export RL3_TCOND=$TC RL3_FREE_LAUNCH=1 RL3_KEEP_LAUNCH=1 RL3_FASTWIND=1 RL3_NEWENT=0.35 \
  SLMODEL=${SLMODEL:-$T/rl2_sl_116668800.npz} TCOND_RUN=$1
RUN=$1; NUPD=$2; shift 2
nice -n 10 ./.venv-gpu/bin/python dev/async_ppo.py learn --run $RUN --launch --local ${LOCAL:-4} \
  --batch ${BATCH:-256} --ckpt_every 0 "$@" > $RUN.log 2>&1 &
L=$!
t0=$(date +%s)
while kill -0 $L 2>/dev/null && [ $(grep -c '^v[0-9]' $RUN.log) -lt $NUPD ] && [ $(( $(date +%s) - t0 )) -lt ${LIMIT:-1500} ]; do sleep 5; done
kill -INT $L 2>/dev/null; sleep 3; kill $L 2>/dev/null
# the workers (and their forkserver children, whose command line does not name the run) carry TCOND_RUN
for sig in TERM KILL; do
  for e in /proc/[0-9]*/environ; do tr '\0' '\n' 2>/dev/null < $e | grep -qx "TCOND_RUN=$RUN" && kill -$sig $(basename $(dirname $e)) 2>/dev/null; done
  sleep 2
done
wait $L 2>/dev/null
echo "elapsed $(( $(date +%s) - t0 )) s"; grep -v "^\s*$" $RUN.log | grep -v Warning | tail -14

#!/usr/bin/env bash
# D: a local CPU learner (+ LOCAL workers, default 2) on scenario 4 (seeds 10000-10003, worlds reused) with the s4arrv line's
# env until NUPD PPO updates are done (or LIMIT s), then stopped.   dev/tc2_check/mini_train.sh RUN NUPD [learner args...]
# (RL3_TCOND2 from the caller)
source "$(dirname "$0")/env.sh"; cd $W
export CUDA_VISIBLE_DEVICES= RL3_FREE_LAUNCH=1 RL3_KEEP_LAUNCH=1 RL3_WAIT_NODISCOUNT=1 RL3_NEWENT=0.35 RL3_FASTWIND=1 \
  SIL_MIN=${SIL_MIN:-1} RL3_REUSE=${RL3_REUSE:-3} SEED_LIST="10000 10001 10002 10003" SLMODEL=$SL8
RUN=$1; NUPD=$2; shift 2
# the learner, its workers and their forkserver children carry TC2_RUN (this script does not, so it is not killed)
TC2_RUN=$RUN nice -n 10 ./.venv-gpu/bin/python dev/async_ppo.py learn --run $RUN --launch \
  --local ${LOCAL:-2} --batch ${BATCH:-128} --ckpt_every 0 "$@" > $RUN.log 2>&1 &
L=$!
t0=$(date +%s)
while kill -0 $L 2>/dev/null && [ $(grep -c '^v[0-9]' $RUN.log) -lt $NUPD ] && [ $(( $(date +%s) - t0 )) -lt ${LIMIT:-540} ]; do sleep 5; done
kill -INT $L 2>/dev/null; sleep 3; kill $L 2>/dev/null
for sig in TERM KILL; do
  for e in /proc/[0-9]*/environ; do tr '\0' '\n' 2>/dev/null < $e | grep -qx "TC2_RUN=$RUN" && kill -$sig $(basename $(dirname $e)) 2>/dev/null; done
  sleep 2
done
wait $L 2>/dev/null
echo "elapsed $(( $(date +%s) - t0 )) s"; grep -v "^\s*$" $RUN.log | grep -v -i warn | grep -E "^v[0-9]|resumed|initialised|added|expanded|free launch|Error|error" | tail -14

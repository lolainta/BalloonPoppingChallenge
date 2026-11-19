#!/usr/bin/env bash
# The local machine as extra async workers for a live cluster run (cp18 is reachable from here, and the worker auth
# key ~/.bpc_remote_key matches the cluster's). learner.addr, the live policy and the SL model are mirrored from the
# cluster every 15 s (workers reload on mtime change); no decision logs are written here (SL trains on the cluster).
#   setsid nohup dev/local_workers.sh late3 >> logs/local_workers.log 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
run=${1:-late3}; N=${N:-16}; R='user@login:~/bpc'
mkdir -p runs/$run
sync_once() { rsync -az --include='live_policy.npz' --include='learner.addr' --include='rl2_sl.npz' --include='seeds.txt' --exclude='*' "$R/runs/$run/" runs/$run/; }
sync_once
( while true; do sleep 15; sync_once; done ) &
ENVS=$(ssh -o BatchMode=yes user@login "cat ~/bpc/runs/$run/worker_env")
export $(echo "$ENVS" | tr ',' '\n' | grep -v '^T1DATA=')  # same env as the run's cluster workers, minus decision logs
echo "$(date +%T) $N local workers for $run ($(cat runs/$run/learner.addr))"
exec nice -n 10 .venv/bin/python dev/async_ppo.py worker --run runs/$run --n $N --launch

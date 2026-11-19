#!/usr/bin/env bash
# Deterministic seed hunt on this host: POLICY (+ its paired SL) on a comma-separated seed list, 96 seeds per chunk.
# Each chunk -> logs/eval_<TAG>_<i>.log in the eval-job format that dev/candidates.py collects, verifies and
# (via the dashboard) records videos for.
#   setsid nohup dev/local_seed_hunt.sh POLICY SEEDFILE TAG [extra k=v ...] >> logs/seed_hunt.log 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
P=$1; SF=$2; TAG=$3; shift 3; EXTRA="$*"
IFS=, read -ra S < "$SF"; n=${#S[@]}; i=0
[ -f logs/hunt_$TAG.meta ] || printf '{"tag": "%s", "policy": "%s", "extra": "%s", "total": %d, "started": %d}\n' \
  "$TAG" "$P" "$EXTRA" $n $(date +%s) > logs/hunt_$TAG.meta  # read by the dashboard's local card
while [ $((i * 96)) -lt $n ]; do
  chunk=$(IFS=,; echo "${S[*]:$((i * 96)):96}")
  f=logs/eval_${TAG}_$i.log
  if [ ! -s $f ]; then
    kw="policy_path=$P sl_path=${P/rl3_policy/rl2_sl} $EXTRA"
    out=$(AGENT=rl3 nice -n 19 .venv/bin/python dev/run_seeds.py --seeds $chunk --kw $kw 2>/dev/null)  # lowest priority: evals first
    { echo "variant: $kw"; echo "agent rl3"; echo "$out"; } > $f
    echo "$(date +%T) chunk $i: $(echo "$out" | awk '/^seed/{n++; if($4>=11)e++; if($4>=12)t++; if($4>M){M=$4; b=$2" @ "$6"s"}} END{printf "%d seeds, >=11: %d, >=12: %d, best %d (%s)", n, e+0, t+0, M, b}')"
  fi
  i=$((i + 1))
done
echo "$(date +%T) done: $n seeds"

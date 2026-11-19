#!/usr/bin/env bash
# Run dev/line_cmp.sh when RUN passes FORK + each offset (steps): [TAG=line] dev/line_sched.sh <run> <base npz> <fork step> <seed0> <offsets...>
# -> logs/<TAG>_<run>_<+offset>.log (use a new TAG for each change on the same run, or the logs are overwritten)
cd "$(dirname "$0")/.."
RUN=$1; BASE=$2; FORK=$3; S0=$4; shift 4
latest() { ssh -o BatchMode=yes user@login "ls ~/bpc/runs/$RUN/rl3_policy_*.npz" 2>/dev/null | sed 's/.*_//; s/.npz//' | sort -n | tail -1; }
for off in "$@"; do
  until [ "$(latest)" -ge $((FORK + off)) ] 2>/dev/null; do sleep 300; done
  lab="+$(awk -v o=$off 'BEGIN{printf "%.1fM", o/1e6}')"
  dev/line_cmp.sh $RUN $BASE "$lab" $S0 ${NSEEDS:-384} > logs/${TAG:-line}_${RUN}_${lab#+}.log 2>&1
  S0=$((S0 + 400))
done

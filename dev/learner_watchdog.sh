#!/usr/bin/env bash
# Restart a stalled learner (runs on login02): if RUN's learner job is RUNNING but its status.json has not been updated
# for STALL seconds, cancel it and resubmit it with <run>/learner_submit.sh (resumes from async_last.pt; the file holds
# the exact sbatch line — sacct's SubmitLine is not usable, the accounting DB is sometimes down).
# 2026-09-29: a learner hung 1.5 h in recv() on a dead worker connection.   setsid nohup dev/learner_watchdog.sh runs/s4ent >> logs/learner_watchdog.log 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
# 2026-11-04: STALL 1200 -> 3600 s. status.json is written once per PPO update, and with ~25 workers one update took 830 s
# (s40l); a restart gives the GPU away to other users' queued jobs (DECISION_LOG §452: 10 h before the learners got one back).
RUN=$1; STALL=${2:-3600}; tag=$(basename $RUN)
while true; do
  sleep 300
  q=$(squeue -u $USER -h -n bpc-async -t RUNNING,PENDING -o "%i %k %T %r" 2>/tmp/llt_wd_err.$$) || continue  # squeue itself fails ~13% of the time (DNS)
  # 2026-10-08: during a DNS outage squeue printed getaddrinfo errors, exited 0 and listed nothing -> "learner gone" twice ->
  # a second learner was submitted beside the running one: any message on stderr = no answer this round
  [ -s /tmp/llt_wd_err.$$ ] && continue
  if ! echo "$q" | awk -v t=$tag '$2 == t {f=1} END {exit !f}'; then
    # 2026-10-01: the learner hit its 40 h time limit and nothing resubmitted it. Job gone on 2 checks in a row -> resubmit,
    # unless <run>/learner_stop exists (touch it before stopping a learner on purpose).
    gone=$(( ${gone:-0} + 1 ))
    if [ $gone -ge 2 ] && [ ! -e $RUN/learner_stop ] && [ -s $RUN/learner_submit.sh ]; then
      echo "$(date '+%F %T') $tag: no learner job -> resubmit with $RUN/learner_submit.sh"; bash $RUN/learner_submit.sh; gone=0
    fi
    continue
  fi
  gone=0
  # 2026-10-02: after a Slurm control-plane outage the learner came back PENDING launch_failed_requeued_held (priority 0)
  # and sat there 2.5 h while this loop counted it as present -> release a held learner
  # (squeue prints the reason with spaces, "launch failed requeued held": match the whole line, not field 4 -- the first
  # version matched $4 = "launch" and let the learner sit held for 33 h, 10-04 01:04 .. 10-05 10:34)
  h=$(echo "$q" | awk -v t=$tag '$2 == t && $3 == "PENDING" && tolower($0) ~ /held/ {print $1}' | head -1)
  # 2026-10-05: the workers of the run come back held too and the keeper counts them as present -> release them as well
  w=$(squeue -u $USER -h -t PENDING -o "%i %k %r" 2>/dev/null | awk -v t=$tag 'index($2, t "@") == 1 && tolower($0) ~ /held/ {print $1}')
  [ -z "$w" ] || { echo "$(date '+%F %T') $tag: held worker jobs $(echo $w | wc -w) -> scontrol release"; scontrol release $w; }
  if [ -n "$h" ]; then echo "$(date '+%F %T') $tag: learner job $h held -> scontrol release"; scontrol release $h; continue; fi
  j=$(echo "$q" | awk -v t=$tag '$2 == t && $3 == "RUNNING" {print $1}' | head -1)
  [ -n "$j" ] || continue
  st=$RUN/status.json; [ -f $st ] || continue
  age=$(( $(date +%s) - $(stat -c %Y $st) ))
  started=$(date -d "$(squeue -h -j $j -o %S)" +%s 2>/dev/null || echo 0)
  [ $(( $(date +%s) - started )) -gt $STALL ] || continue  # a fresh learner needs time for its first update
  if [ $age -gt $STALL ]; then
    sub=$RUN/learner_submit.sh
    [ -s $sub ] || { echo "$(date '+%F %T') $tag job $j stalled ${age}s but $sub is missing: not restarting"; continue; }
    # 2026-11-04: a successor queued with --dependency=afterany:$j takes over once $j is cancelled; resubmitting as well
    # would put two learners on the same run
    n=$(echo "$q" | awk -v t=$tag -v j=$j '$2 == t && $1 != j' | wc -l)
    echo "$(date '+%F %T') $tag job $j: status.json ${age}s old -> cancel, $([ $n -gt 0 ] && echo "$n queued successor(s) take over" || echo "restart with $sub")"
    scancel $j; sleep 20
    [ $n -gt 0 ] || bash $sub
  fi
done

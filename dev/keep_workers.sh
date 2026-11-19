#!/usr/bin/env bash
# Keep async env workers alive: they run on the low-priority QoS, yield to anyone's normal jobs (devq_low is
# preempted with CANCEL) and get resubmitted here once their node is free again. Runs on the login node.
# Until NORMAL_UNTIL (user decision 2026-09-25 01:30: "normal QoS until 10:00") everything goes on the normal QoS;
# at that time our normal-QoS workers / teachers are cancelled once and come back on devq_low.
#   nohup dev/keep_workers.sh runs/workers.spec > logs/keep_workers.log 2>&1 &
# spec lines:  RUN NODE N MEM ENV   e.g.  runs/ppo8 cp17 31 48G T1DATA=...,SLMODEL=...,RL3_T_MIN=1
spec=${1:-runs/workers.spec}
NORMAL_UNTIL=${NORMAL_UNTIL:-2026-09-25 10:00}
cd ~/bpc
# WORKER_TIME (2026-10-14): short worker jobs can backfill cores Slurm holds for someone's pending job (a 40 h job never fits)
# BACKFILL_NODES: only these get the short WORKER_TIME (nodes Slurm plans for someone's pending job); elsewhere 40 h, so a
# worker does not stop every 2 h for nothing (cp08 sat idle ~2 min at each rollover, 2026-10-15)
time_of() { case " ${BACKFILL_NODES:-} " in *" $1 "*) echo ${WORKER_TIME:-40:00:00} ;; *) echo 40:00:00 ;; esac; }
qos_of() { case " ${QOS_LOW_NODES:-} " in *" $1 "*) echo devq_low ;; *) echo $QOS ;; esac; }
while true; do
  # QOS_LOW_NODES: nodes that keep the low-priority QoS even while normal applies (user 2026-09-29: cp14 at most devq_low)
  if [ $(date +%s) -lt $(date -d "$NORMAL_UNTIL" +%s) ]; then QOS=normal; else
    QOS=devq_low
    if [ ! -f runs/.qos_back_to_low ]; then  # once: move our normal-QoS fillers back to the low-priority QoS
      touch runs/.qos_back_to_low
      squeue -h -u $USER -o "%i %j %q" | awk '($2 == "bpc-aworker" || $2 == "bpc-teach") && $3 == "normal" {print $1}' | xargs -r scancel
      echo "$(date +%T) $NORMAL_UNTIL passed: workers / teachers back on devq_low"; sleep 15
    fi
  fi
  # a worker stuck as held / launch-failed never runs: cancel it so it is resubmitted below
  squeue -h -u $USER -n bpc-aworker -t PD -o "%i %r" | awk '/held|launch_failed/{print $1}' | xargs -r scancel
  # a worker / teacher waiting for cores someone else took (e.g. preempted, then half the node was grabbed) never
  # starts and blocks the other on that node: cancel it, it is resubmitted below when (or where) it fits
  squeue -h -u $USER -t PD -o "%i %j %r %k" | awk '($2 == "bpc-aworker" || $2 == "bpc-teach") && $4 !~ /~fill$/ && /Resources|ReqNodeNotAvail|reserved/ {print $1}' | xargs -r scancel
  # a running worker whose node shows almost no CPU load is hung (cp13 / cp16, 2026-09-25: the env pool's forkserver
  # blocks in an NFS read): after 5 checks in a row (~5 min) cancel it, it is resubmitted below
  touch runs/.idle_workers
  squeue -h -u $USER -n bpc-aworker -t R -o "%i %N %C %M" | while read -r j node c el; do
    case "$el" in *:*:*|*-*) ;; [0-9]:*) continue ;; esac  # started < 10 min ago: still spinning up
    load=$(sinfo -h -n $node -o "%O" | cut -d. -f1)
    if [ "${load:-0}" -lt $((c / 4)) ]; then
      n=$(( $(awk -v j=$j '$1 == j {print $2}' runs/.idle_workers) + 1 ))
      if [ $n -ge 5 ]; then scancel $j; echo "$(date +%T) worker $j on $node idle (load $load / $c cores): cancelled"; n=0; fi
    else n=0; fi
    awk -v j=$j '$1 != j' runs/.idle_workers > runs/.idle_workers.tmp; echo "$j $n" >> runs/.idle_workers.tmp; mv runs/.idle_workers.tmp runs/.idle_workers
  done
  sleep 2
  want=" "  # nodes that just got a worker submitted: sinfo may not show it yet, so no teacher filler there this round
  while read -r run node n mem env; do
    [ -z "$run" ] || [ "${run:0:1}" = "#" ] && continue
    tag="$(basename $run)@$node"
    if ! squeue -h -u $USER -n bpc-aworker -o "%k" | grep -qx "$tag"; then
      free=$(sinfo -h -n $node -o "%C" | cut -d/ -f2)
      tj=$(squeue -h -u $USER -n bpc-teach -w $node -o "%i %C" 2>/dev/null)  # our teacher fillers on this node
      tc=$(echo "$tj" | awk '{s += $2} END {print s + 0}')
      if [ "${free:-0}" -lt $((n + 1)) ] && [ $((${free:-0} + tc)) -ge $((n + 1)) ]; then
        # workers first: only our own teacher fillers keep the worker out -> make room, it goes in next round
        echo "$tj" | awk '{print $1}' | xargs -r scancel; want="$want$node "
        echo "$(date +%T) $tag: cancelled teacher fillers on $node to make room"
      fi
      fmb=$(sinfo -h -n $node -O "memory:12,allocmem:12" | awk '{print $1 - $2}')  # free MB: 2026-10-20 cp15, someone
      case $mem in *G) need=$(( ${mem%G} * 1024 )) ;; *) need=0 ;; esac          # held 48 of 63 GB -> our 24 GB worker
      if [ "${free:-0}" -ge $((n + 1)) ] && [ "${fmb:-0}" -ge "$need" ]; then        # pended, cancelled, resubmitted each minute
        want="$want$node "
        sbatch -q $(qos_of $node) --time=$(time_of $node) --comment="$tag" --export=ALL,RUN=$run,N=$n,LAUNCH=1,$env -w $node -c $((n + 1)) --mem=$mem dev/async_worker.sbatch \
          | sed "s/^/$(date +%T) $tag: /"
      elif [ -z "$(cat runs/teach_run 2>/dev/null)" ] && [ "${free:-0}" -ge 4 ]; then
        # no teacher fillers: someone else holds part of the node, so run a smaller worker on what is left
        fm=$(sinfo -h -n $node -O "memory:12,allocmem:12" | awk '{print int(($1 - $2) / 1500)}')  # (2026-10-16: cp20 24 cores
        [ "${fm:-0}" -lt "$free" ] && free=$fm                                                     # asked 36 GB of 32 -> rejected)
        [ "$free" -ge 4 ] || continue
        want="$want$node "; m=$((free - 1))
        sbatch -q $(qos_of $node) --time=$(time_of $node) --comment="$tag" --export=ALL,RUN=$run,N=$m,LAUNCH=1,$env -w $node -c $free --mem=$((free * 1500))M dev/async_worker.sbatch \
          | sed "s/^/$(date +%T) $tag (partial node, $m envs): /"
      fi
    fi
  done < "$spec"
  # idle cores on any node of the spec (someone left a gap, Slurm plans the node for a queued job, our main worker
  # waits): many small low-priority workers (user 2026-10-16: "place many devq_low small jobs with a few worker to
  # maximize the total usage"): FILL_CORES each (FILL_CORES - 1 envs), FILL_TIME limit (short: fits gaps before planned
  # jobs), devq_low, comment "<run>@<node>~fill" (handovers that cancel "<run>@..." include them); no new ones on a
  # node while a filler there is pending, at most FILL_MAX per node and pass
  if [ -z "$(cat runs/teach_run 2>/dev/null)" ] && pend=$(squeue -h -u $USER -n bpc-aworker -t PENDING -o "%k" 2>/dev/null); then
    FILL=${FILL_CORES:-4}
    while read -r run node n mem env; do
      [ -z "$run" ] || [ "${run:0:1}" = "#" ] && continue
      case "$want" in *" $node "*) continue ;; esac
      case " ${NO_EXTRA_NODES:-} " in *" $node "*) continue ;; esac
      echo "$pend" | grep -q "@$node~fill" && continue  # (only pending fillers hold new ones back; a pending main worker does not)
      free=$(sinfo -h -n $node -o "%C" | cut -d/ -f2)
      fm=$(sinfo -h -n $node -O "memory:12,allocmem:12" | awk '{print int(($1 - $2) / 1500)}')  # (1.5 GB a core, §: cp20)
      [ "${fm:-0}" -lt "${free:-0}" ] && free=$fm
      k=$(( ${free:-0} / FILL )); [ $k -gt ${FILL_MAX:-8} ] && k=${FILL_MAX:-8}
      [ $k -ge 1 ] || continue
      for i in $(seq $k); do
        sbatch -q devq_low --time=${FILL_TIME:-1:00:00} --comment="$(basename $run)@$node~fill" \
          --export=ALL,RUN=$run,N=$((FILL - 1)),LAUNCH=1,$env -w $node -c $FILL --mem=$((FILL * 1500))M dev/async_worker.sbatch > /dev/null
      done
      echo "$(date +%T) $(basename $run)@$node: $k fillers of $((FILL - 1)) envs ($free idle cores)"
    done < "$spec"
  fi
  # idle leftovers (e.g. someone else holds half a node, so a full worker never fits): fill them with teacher
  # search jobs for the run named in runs/teach_run (dev/search_teacher.sbatch); no file = no fillers
  tr=$(cat runs/teach_run 2>/dev/null)
  if [ -n "$tr" ] && ls runs/$tr/rl3_policy_[0-9]*.npz >/dev/null 2>&1 && [ -z "$(squeue -h -u $USER -n bpc-teach -t PD)" ]; then
    sleep 5  # (only once the run has a checkpoint, and none still queued)
    sinfo -h -N -o "%N %C %T" | awk "\$3 !~ /down|drain|resv/" | while read -r node c st; do
      free=$(echo $c | cut -d/ -f2)
      case "$want" in *" $node "*) continue ;; esac
      if [ "${free:-0}" -ge 8 ]; then
        sbatch -q $QOS -w $node -c $free --mem=$((free * 2))G --comment=teach@$tr --export=ALL,RUN=$tr,NSEEDS=$((free * 12)) dev/search_teacher.sbatch \
          | sed "s/^/$(date +%T) teacher@$node ($free cores): /"
      fi
    done
  fi
  # node-local python copies (/tmp/llt_py, /tmp/llt_gpu; DECISION_LOG §229): systemd-tmpfiles deletes /tmp entries older
  # than 10 days every day (17:49), and this tmpfs keeps no fresh atime, so a copy made 10 days ago lost most of its files
  # even while in use (cp11 workers then failed in 2 s, cp16's learner would not have restarted) -> twice a day refresh the
  # copies' access / change times on every node we hold a job on (touch -a: mtime unchanged, cached .pyc stay valid)
  if [ $(( $(date +%s) - ${last_touch:-0} )) -gt 43200 ]; then
    last_touch=$(date +%s)
    squeue -h -u $USER -t R -o "%i %N" | sort -k2 -u | { while read -r j n; do
      timeout 300 srun --overlap --jobid=$j -n1 -c1 bash -c 'find /tmp/llt_py /tmp/llt_gpu -mindepth 1 -exec touch -a -c -h {} + 2>/dev/null; true' &
    done; wait; }
    echo "$(date +%T) refreshed the /tmp copies' times"
  fi
  sleep 60
done

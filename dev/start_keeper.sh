#!/usr/bin/env bash
# start the worker keeper (user 2026-10-16, 10-19): every worker on devq_low (user 2026-10-19: "for worker, use devq_low";
# NORMAL_UNTIL in the past; GPU users on cp18 preempt our devq_low workers when they need it, user 2026-10-19), RTX 4090 nodes (cp14 cp15 cp17 cp20) with short (1 h) jobs, the others 40 h; idle cores anywhere filled with small devq_low workers (FILL_CORES, FILL_TIME), see keep_workers.sh
cd ~/bpc
NORMAL_UNTIL="2026-10-19 00:00" QOS_LOW_NODES="cp14 cp15 cp17 cp20" WORKER_TIME=1:00:00 BACKFILL_NODES="cp14 cp15 cp17 cp20" \
  FILL_CORES=4 FILL_TIME=1:00:00 FILL_MAX=8 setsid nohup dev/keep_workers.sh runs/workers01.spec >> logs/keep_workers.log 2>&1 < /dev/null &

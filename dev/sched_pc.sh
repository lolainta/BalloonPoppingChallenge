#!/usr/bin/env bash
# DECISION_LOG §511 scheduler on login (user 11/8 00:58: "你來做"):  setsid nohup bash dev/sched_pc.sh >> logs/sched_pc.log 2>&1 < /dev/null &
# Waits for the 64 confirmation drivers (dev/launch_pc.sh), pools them with the §509 rule (ONLY1=solo), and only if the
# VERDICT names cfg1 = pqp (and it is not past 11:30) builds the single file and runs the official 16/16 (dev/pack_pc.sh),
# then waits for that check and logs its result. Missing set-hours stop it before any VERDICT (rerun them, then pool).
set -u
cd ~/bpc
t() { date -d "2026-11-08 $1" +%s; }
echo "$(date '+%F %T') sched_pc waiting for the probe drivers"
while pgrep -u $USER -f 'dev/probe_c[m]p.sh' > /dev/null; do sleep 60; done
echo "$(date +%T) drivers done ($(ls runs/.cmp_parts/probe_pc?/hour*.csv 2>/dev/null | wc -l)/64 set-hours); pooling"
rm -f logs/probe_pc_all.log
P=pc S0=10067000 W1=pqp W2=pql ONLY1=solo BN=phe BF=bpc_s40_fin_phe.py bash dev/pool_phc.sh 2>&1 | tail -n 12
grep -q '^VERDICT:' logs/probe_pc_all.log 2>/dev/null || { echo "$(date +%T) STOP: pooling did not finish (missing set-hours?)"; exit 1; }
grep -q '^VERDICT: cfg1 replaces phe -> W=pqp ' logs/probe_pc_all.log || { echo "$(date +%T) DONE: no pass, the finale keeps phe"; exit 0; }
[ $(date +%s) -lt $(t 11:30) ] || { echo "$(date +%T) DONE: passed but past 11:30, no packing, the finale keeps phe"; exit 0; }
echo "$(date +%T) packing pqp"
out=$(bash dev/pack_pc.sh 2>&1); echo "$out"
J=$(echo "$out" | grep -oP 'Submitted batch job \K[0-9]+') || { echo "$(date +%T) STOP: pack failed, the finale keeps phe"; exit 1; }
while squeue -h -j $J 2>/dev/null | grep -q .; do sleep 60; done
echo "$(date +%T) DONE: offeval $J: $(grep -E 'identical' logs/offeval_$J.log)  file $(md5sum BalloonPoppingGymEnv/agents/bpc_s40_fin_pqp.py)"

#!/usr/bin/env bash
# DECISION_LOG §502 Phase F scheduler on login:  setsid nohup bash dev/sched_phf.sh >> logs/sched_phf.log 2>&1 < /dev/null &
# From 01:45 11/8, every 5 min: launch dev/launch_phf.sh once its DRY check passes (every branch >= 48 exports after phe's
# window); not ready by 03:30 -> give up (finale keeps phe). Then wait for the 8 probe drivers, pool, and only if the VERDICT
# names a config (and it is not past 11:30) pack it and run the official 16/16 check on 10065100-10065115.
set -u
cd ~/bpc
t() { date -d "2026-11-08 $1" +%s; }
echo "$(date '+%F %T') sched_phf waiting for 01:45"
while [ $(date +%s) -lt $(t 01:45) ]; do sleep 60; done
until DRY=1 bash dev/launch_phf.sh > /dev/null 2>&1; do
  if [ $(date +%s) -ge $(t 03:30) ]; then echo "$(date +%T) GIVE UP: not ready by 03:30; finale keeps phe"; DRY=1 bash dev/launch_phf.sh; exit 1; fi
  sleep 300
done
echo "$(date +%T) launching Phase F"
bash dev/launch_phf.sh || { echo "$(date +%T) launch failed; finale keeps phe"; exit 1; }
sleep 120
while pgrep -u $USER -f 'dev/probe_c[m]p.sh' > /dev/null; do sleep 120; done
echo "$(date +%T) all probe drivers done; pooling"
P=phf S0=10062000 W1=phf W2=phf2 BN=phe BF=bpc_s40_fin_phe.py bash dev/pool_phc.sh 2>&1 | tail -n 8
W=$(grep -oP '^VERDICT: cfg[12] replaces phe -> W=\K\S+' logs/probe_phf_all.log 2>/dev/null) || { echo "$(date +%T) no winner: finale keeps phe"; exit 0; }
[ $(date +%s) -lt $(t 11:30) ] || { echo "$(date +%T) $W won but it is past 11:30: no packing, finale keeps phe"; exit 0; }
echo "$(date +%T) packing $W"
out=$(OFF_S0=10065100 BN=phe W=$W P=phf bash dev/pack_phc.sh 2>&1); echo "$out"
J=$(echo "$out" | grep -oP 'Submitted batch job \K[0-9]+') || { echo "$(date +%T) pack failed: finale keeps phe"; exit 1; }
while squeue -h -j $J 2>/dev/null | grep -q .; do sleep 60; done
echo "$(date +%T) offeval $J: $(grep -E 'identical' logs/offeval_$J.log)  file $(md5sum BalloonPoppingGymEnv/agents/bpc_s40_fin_$W.py)"

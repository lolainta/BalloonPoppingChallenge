#!/bin/bash
# Hybrid E2E M3 (DECISION_LOG §262): every new RUN/net_<v>.pt -> a 16-core normal job flying it (mean corrections, E2E_RES=1) on
# seeds LO..HI (launch hours cycling 0~21) -> logs/e2e_reseval_<run>_<v>.log; pair with the teacher's log (evalnet teacher).
#   (login) setsid nohup bash dev/e2e_eval_sched.sh runs/e2e_res1 10001000 10001095 > logs/e2e_eval_sched.log 2>&1 &
cd ~/bpc
RUN=$1 LO=$2 HI=$3 NODE=${NODE:-cp15}
KW="${E2E_KW:-policy_path=runs/s40a/rl3_policy_391874515.npz sl_path=runs/s40a/rl2_sl_391874515.npz att_fix=0.07}"
while true; do
  for f in $RUN/net_*.pt; do
    [ -e "$f" ] || continue
    v=${f##*net_}; v=${v%.pt}; out=logs/e2e_reseval_$(basename $RUN)_$v.log
    [ -e "$out" ] && continue
    echo "$(date +%T) submit $f"
    sbatch -q normal -p devq -w $NODE -c 16 --mem=24G --time=1:00:00 -J bpc-e2eev -o $out \
      --wrap "cd ~/bpc; E2E_RES=1 E2E_KW=\"$KW\" PROCS=16 /tmp/llt_py/.venv/bin/python dev/e2e.py evalnet $f $LO $HI"
  done
  sleep 300
done

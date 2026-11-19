#!/usr/bin/env bash
# Held-out eval (deterministic, seeds 10000-10095) of every live learner's newest checkpoint, run on the cluster
# (user 2026-11-03: no BPC compute on the local machine; this replaces the eval half of dev/local_services.sh).
# One 16-core normal-QoS job per run per pass, a pass every hour (training keeps the cores in between);
# results -> runs/<run>/det_eval.csv + det_eval_<step>.txt, which dev/local_services.sh mirrors for the dashboard.
#   setsid nohup dev/det_eval_cluster.sh >> logs/det_eval_cluster.log 2>&1 < /dev/null &
cd ~/bpc
while true; do
  for run in $(squeue -h -u $USER -n bpc-async -t R -o %k); do
    case $run in s40*) sc=40 ;; s4*) sc=4 ;; *) continue ;; esac
    f=$(ls runs/$run/rl3_policy_[0-9]*.npz 2>/dev/null | sort -t_ -k3 -n | tail -1); [ -n "$f" ] || continue
    step=$(basename $f .npz | sed 's/.*_//'); sl=runs/$run/rl2_sl_$step.npz; csvf=runs/$run/det_eval.csv
    [ -f $sl ] || continue  # never evaluate a checkpoint without its SL
    [ -f $csvf ] || echo "step,mean,min,max,ge8,ge9" > $csvf
    grep -q "^$step," $csvf && continue
    ( jl=runs/$run/det_eval_$step.log
      sbatch --wait -q normal --exclude=${EXCLUDE:-cp11,cp14,cp18} -c 16 --mem=20G --time=0:40:00 -J bpc-deteval -o $jl \
        --wrap "PY=/tmp/llt_py/.venv/bin/python; . dev/node_boot.sh; AGENT=rl3 \$PY dev/run_seeds.py --seeds 10000-10095 --scenario $sc --kw policy_path=$f sl_path=$sl 2>/dev/null" > /dev/null
      out=$(grep -a "^seed" $jl 2>/dev/null)
      [ -n "$out" ] || { echo "$(date +%T) $f: no results (see $jl)"; exit; }
      { echo "kw policy_path=$f sl_path=$sl"; echo "agent rl3"; echo "scenario $sc"; cat $jl; } > runs/$run/det_eval_$step.txt; rm -f $jl
      echo "$out" | awk -v s=$step '/^seed/{n++; t+=$4; if(n==1||$4<m)m=$4; if($4>M)M=$4; if($4>=8)a++; if($4>=9)b++} END{if(n) printf "%s,%.2f,%d,%d,%d,%d\n", s, t/n, m, M, a+0, b+0}' >> $csvf
      echo "$(date +%T) evaluated $f: $(tail -1 $csvf)" ) &
  done; wait
  sleep 3600
done

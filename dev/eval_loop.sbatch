#!/bin/bash
# Deterministic held-out eval (seeds 10000-10095) of every new checkpoint of a run -> $RUN/det_eval.csv
#SBATCH -p devq -J bpc-evalloop -c 16 --mem=32G --time=40:00:00 --nice=1000
#SBATCH -o logs/eval_loop_%j.log
cd ~/bpc
RUN=${RUN:-runs/ppo5}; AG=${AG:-rl2}; PAT=${PAT:-rl2_policy_[0-9]*.npz}
[ -f $RUN/det_eval.csv ] || echo "step,mean,min,max,ge8,ge9" > $RUN/det_eval.csv
while true; do
  for f in $(ls $RUN/$PAT 2>/dev/null | sort -t_ -k3 -n | tail -1); do  # newest checkpoint only: never fall behind
    step=$(basename $f .npz | sed 's/.*_//')
    grep -q "^$step," $RUN/det_eval.csv && continue
    sl=$RUN/rl2_sl_$step.npz  # SL model live at checkpoint time (trainer writes it; else newest stamped copy older than $f)
    if [ ! -f $sl ]; then
      c=$(for x in $RUN/rl2_sl_t*.npz; do [ -f "$x" ] && [ ! "$x" -nt "$f" ] && echo "$x"; done | sort | tail -1)
      [ -n "$c" ] && cp -p "$c" $sl
    fi
    slkw="use_sl=0"; [ -f $sl ] && slkw="sl_path=$sl"
    out=$(AGENT=$AG ${PY:-.venv/bin/python} dev/run_seeds.py --seeds 10000-10095 ${SC:+--scenario $SC} --kw policy_path=$f $slkw 2>/dev/null)  # PY: a node-local python (§: never from NFS on compute nodes); SC: scenario
    { echo "kw policy_path=$f $slkw"; echo "agent $AG"; echo "$out"; } > $RUN/det_eval_$step.txt  # exact kwargs + per-seed results
    echo "$out" | awk -v s=$step '/^seed/{n++; t+=$4; if(n==1||$4<m)m=$4; if($4>M)M=$4; if($4>=8)a++; if($4>=9)b++} END{if(n) printf "%s,%.2f,%d,%d,%d,%d\n", s, t/n, m, M, a+0, b+0}' >> $RUN/det_eval.csv
    echo "$(date +%T) evaluated $f"
  done
  sleep 120
done

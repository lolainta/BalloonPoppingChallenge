#!/usr/bin/env bash
# Zero-training probe on login (DECISION_LOG §357): one BASE checkpoint flown with CFGS overrides (agent kwargs, e.g.
# "max_rate=3.0;k_att=3.6" or "policy_path=runs/s40l/avg5_X.npz") on fresh paired seeds, one 1-hour normal-QoS job per
# launch hour (dev/line_cmp.sbatch CFGS mode, §319 layout), pooled into logs/probe_<TAG>.log.
#   TAG=avg HOURS="0 6 12 18" S0=10027400 BASE=runs/s40l/rl3_policy_X.npz CFGS="policy_path=...;policy_path=..." \
#     setsid nohup bash dev/probe_cmp.sh >> logs/probe_sched.log 2>&1 < /dev/null &
# Seeds: S0 + 96 per hour (register them in DECISION_LOG; sweep_gnc.py records them in the seed registry).
set -u
cd ~/bpc
d=runs/.cmp_parts/probe_$TAG; mkdir -p $d; log=logs/probe_${TAG}.log
hrs=(${HOURS:-0 6 12 18}); m=${N:-96}; i=0
export CFGS  # (reaches the job through --export=ALL: a value in the --export list is cut at its commas, e.g. ens=a,b)
for h in "${hrs[@]}"; do
  ( for try in 1 2; do  # one retry if the hour left no CSV
      sbatch --wait -q normal --exclude=${EXCLUDE:-cp14,cp18} -c ${CORES:-16} --mem=${MEM:-20G} --time=${TIME:-1:00:00} -o $d/hour$h.log \
        --export=ALL,RUN=${RUN:-s40l},BASE=$BASE,LABEL=probe_$TAG,S0=$((S0 + i * m)),N=$m,HOURS=$h,OUT_CSV=$d/hour$h.csv,SC=${SC:-40},BASE_EXTRA=${BASE_EXTRA:-att_fix=0.07} dev/line_cmp.sbatch > /dev/null
      [ -s $d/hour$h.csv ] && break; done ) &
  i=$((i + 1))
done; wait
{ for h in "${hrs[@]}"; do head -1 $d/hour$h.log; done
  .venv/bin/python dev/pool_csv.py ${SC:-40} $(for h in "${hrs[@]}"; do echo hour$h=$d/hour$h.csv; done)
  .venv/bin/python dev/seed_backfill.py ${SC:-40} probe_$TAG $(i=0; for h in "${hrs[@]}"; do echo $((S0 + i * m))-$((S0 + i * m + m - 1)); i=$((i + 1)); done); } > $log 2>&1
echo "$(date +%T) probe $TAG pooled -> $log"

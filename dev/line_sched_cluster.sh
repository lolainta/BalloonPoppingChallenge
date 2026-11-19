#!/usr/bin/env bash
# dev/line_sched.sh on login (user 2026-10-19: no BPC compute on the local host; DECISION_LOG §252): when RUN passes
# FORK + each offset, run dev/line_cmp.sbatch on NODE (normal QoS) and wait; SC / HOURS / BASE_EXTRA pass through.
# (§285: no NODE -> Slurm picks a node outside EXCLUDE (cp14 / cp18: the user's placement rule) and preempts our devq_low
# workers there)
#   TAG=s40a SC=40 HOURS="0 6 12 18" BASE_EXTRA=att_fix=0.07 setsid nohup dev/line_sched_cluster.sh <run> <base npz> <fork step> <seed0> <offsets...>
cd ~/bpc
RUN=$1; BASE=$2; FORK=$3; S0=$4; shift 4
latest() { ls runs/$RUN/rl3_policy_[0-9]*.npz 2>/dev/null | sed 's/.*_//; s/.npz//' | sort -n | tail -1; }
for off in "$@"; do
  until [ "$(latest)" -ge $((FORK + off)) ] 2>/dev/null; do sleep 300; done
  lab="+$(awk -v o=$off 'BEGIN{printf "%.1fM", o/1e6}')"
  # §319: one 1-hour job per launch hour instead of one 6-hour job: a full cluster leaves backfill gaps (our 1 h fillers
  # start, a 6 h normal-QoS job was planned for 03:29 next morning). Same seeds per hour as before; pooled here on login.
  log=logs/${TAG:-line}_${RUN}_${lab#+}.log; d=runs/.cmp_parts/${TAG:-line}_${RUN}_${lab#+}; mkdir -p $d
  hrs=(${HOURS:-0 6 12 18}); m=$((${NSEEDS:-384} / ${#hrs[@]})); i=0
  ck=$(ls runs/$RUN/rl3_policy_[0-9]*.npz | sed 's/.*_//; s/.npz//' | sort -n | tail -2 | paste -sd:)  # §320: pin the two checkpoints now
  for h in "${hrs[@]}"; do  # (MEM 20G: a 16-proc hour peaked at 9.9 GB RSS; 32G no longer fit the nodes' leftover memory)
    ( for try in 1 2; do  # one retry if the hour left no CSV (killed at the time limit, node failure)
        sbatch --wait -q normal ${NODE:+-w $NODE} --exclude=${EXCLUDE:-cp14,cp18} -c ${CORES:-16} --mem=${MEM:-20G} --time=${TIME:-1:00:00} -o $d/hour$h.log \
          --export=ALL,RUN=$RUN,BASE=$BASE,LABEL=$lab,S0=$((S0 + i * m)),N=$m,HOURS=$h,CKPTS=$ck,OUT_CSV=$d/hour$h.csv dev/line_cmp.sbatch > /dev/null
        [ -s $d/hour$h.csv ] && break; done ) &
    i=$((i + 1))
  done; wait
  { for h in "${hrs[@]}"; do head -1 $d/hour$h.log; done
    .venv/bin/python dev/pool_csv.py ${SC:-40} $(for h in "${hrs[@]}"; do echo hour$h=$d/hour$h.csv; done)
    .venv/bin/python dev/seed_backfill.py ${SC:-40} ${TAG:-line}_${RUN}_${lab#+} $(i=0; for h in "${hrs[@]}"; do echo $((S0 + i * m))-$((S0 + i * m + m - 1)); i=$((i + 1)); done); } > $log 2>&1
  S0=$((S0 + 400))
done

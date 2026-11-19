#!/bin/bash
# dev/line_cmp.sh on the cluster (user 2026-10-19: no BPC compute on the local host; DECISION_LOG §252): the two newest
# checkpoints of RUN vs BASE, the seeds split over HOURS (scenario SC), pooled per seed by dev/pool_csv.py -> stdout = the log.
#   sbatch -q normal -w cp13 -c 16 --mem=32G -o logs/<TAG>_<run>_<label>.log \
#     --export=ALL,RUN=s40a,BASE=runs/...npz,LABEL=+2.3M,S0=97200,N=384,SC=40,HOURS="0 6 12 18",BASE_EXTRA=att_fix=0.07 dev/line_cmp.sbatch
#   GNC sweep: --export=ALL,BASE=runs/s40a/rl3_policy_X.npz,LABEL=gnc,S0=...,CFGS="k_att=2.4;max_rate=3" dev/line_cmp.sbatch
#SBATCH -p devq -J bpc-cmp --time=6:00:00
cd ~/bpc
PY=${PY:-/tmp/llt_py/.venv/bin/python}; . dev/node_boot.sh; N=${N:-384}; SC=${SC:-40}; HOURS=${HOURS:-0 6 12 18}  # (node_boot: §285)
cfg=(); for s in ${CKPTS//:/ }; do  # CKPTS=a:b (§320): the scheduler pins the two checkpoints, so hour parts started hours apart agree
  cfg+=("policy_path=runs/$RUN/rl3_policy_$s.npz sl_path=runs/$RUN/rl2_sl_$s.npz ${EXTRA:-}"); done
[ -n "${CKPTS:-}" ] || for s in $(ls runs/$RUN/rl3_policy_[0-9]*.npz | sed 's/.*_//; s/.npz//' | sort -n | tail -2); do
  cfg+=("policy_path=runs/$RUN/rl3_policy_$s.npz sl_path=runs/$RUN/rl2_sl_$s.npz ${EXTRA:-}"); done
[ -n "${CFGS:-}" ] && IFS=';' read -ra cfg <<< "$CFGS"  # CFGS="k_att=2.4;k_att=3.75": these configs on BASE instead (a GNC sweep)
echo "$(date +%T) $LABEL: $RUN ${cfg[*]} vs base $BASE ${BASE_EXTRA:-}, seeds $S0-$((S0 + N - 1)) over hours $HOURS (cluster job $SLURM_JOB_ID)"
set -- $HOURS; m=$((N / $#)); i=0; parts=()
for h in $HOURS; do
  a=$((S0 + i * m)); f=${OUT_CSV:-$(mktemp --suffix=.csv)}; parts+=("hour$h=$f"); i=$((i + 1))  # OUT_CSV (§319): one hour per job, the caller pools
  AGENT=rl3 $PY dev/sweep_gnc.py --scenario $SC --seeds $a-$((a + m - 1)) --procs $SLURM_CPUS_PER_TASK --csv $f \
    --base policy_path=$BASE sl_path=${BASE/rl3_policy/rl2_sl} P.date=2022,7,10,$h ${BASE_EXTRA:-} --cfg "" "${cfg[@]}" > /dev/null 2>&1
done
[ -n "${OUT_CSV:-}" ] || $PY dev/pool_csv.py $SC "${parts[@]}"

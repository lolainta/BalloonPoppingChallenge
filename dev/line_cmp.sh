#!/usr/bin/env bash
# Single-line check (DECISION_LOG §140): the two newest checkpoints of RUN (each with its own saved SL and GNC tunables)
# vs a fixed BASE checkpoint (the line's starting point), paired on fresh scenario-4 seeds (local machine).
#   dev/line_cmp.sh <run> <base policy npz> <label> <first seed> [n seeds]
# scenario 40: SC=40 HOURS="0 6 12 18" BASE_EXTRA="att_fix=0.07" (seeds split over the launch hours, pooled; §248)
cd "$(dirname "$0")/.."
RUN=$1; BASE=$2; label=$3; s0=$4; n=${5:-384}; R='user@login:~/bpc'
mkdir -p runs/$RUN; rsync -az --include='rl3_policy_[0-9]*.npz' --include='rl2_sl_[0-9]*.npz' --exclude='*' $R/runs/$RUN/ runs/$RUN/
cfg=(); for s in $(ls runs/$RUN/rl3_policy_[0-9]*.npz | sed 's/.*_//; s/.npz//' | sort -n | tail -2); do
  cfg+=("policy_path=runs/$RUN/rl3_policy_$s.npz sl_path=runs/$RUN/rl2_sl_$s.npz ${EXTRA:-}"); done  # EXTRA: the run's eval kwargs, e.g. t_min=47
echo "$(date +%T) $label: $RUN ${cfg[*]} vs base $BASE, seeds $s0-$((s0 + n - 1))"
if [ -z "${HOURS:-}" ]; then
AGENT=rl3 .venv/bin/python dev/sweep_gnc.py --scenario ${SC:-4} --seeds $s0-$((s0 + n - 1)) \
  --base policy_path=$BASE sl_path=${BASE/rl3_policy/rl2_sl} ${BASE_EXTRA:-} ${WORLD:-} --cfg "" "${cfg[@]}"  # WORLD: P.<param>=<v> ... (hidden-parameter world, §198)
else  # scenario 40 (§248): HOURS="0 6 12 18" -> the n seeds split over these launch hours, one sweep each, pooled per seed
  set -- $HOURS; m=$((n / $#)); i=0; parts=()  # BASE_EXTRA: kwargs only the base needs (e.g. att_fix=0.07: a checkpoint from
  for h in $HOURS; do                             # before the GNC tunable existed does not carry it)
    a=$((s0 + i * m)); f=$(mktemp --suffix=.csv); parts+=("hour$h=$f"); i=$((i + 1))
    AGENT=rl3 .venv/bin/python dev/sweep_gnc.py --scenario ${SC:-40} --seeds $a-$((a + m - 1)) --csv $f \
      --base policy_path=$BASE sl_path=${BASE/rl3_policy/rl2_sl} P.date=2022,7,10,$h ${BASE_EXTRA:-} ${WORLD:-} --cfg "" "${cfg[@]}" > /dev/null
  done
  .venv/bin/python dev/pool_csv.py ${SC:-40} "${parts[@]}"
fi

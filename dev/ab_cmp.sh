#!/usr/bin/env bash
# A/B at matched steps since the fork (DECISION_LOG §93 protocol): the two newest checkpoints of TREAT and the CTRL
# checkpoints nearest to them, each with its own saved SL and its own GNC tunables (carried in the npz), paired on fresh seeds.
#   dev/ab_cmp.sh <treat run> <ctrl run> <fork step> <label> <first seed> [n seeds]   (local machine)
cd "$(dirname "$0")/.."
TR=$1; CT=$2; FORK=$3; label=$4; s0=$5; n=${6:-384}; R='user@login:~/bpc'
for r in $TR $CT; do mkdir -p runs/$r; rsync -az --include='rl3_policy_[0-9]*.npz' --include='rl2_sl_[0-9]*.npz' --exclude='*' $R/runs/$r/ runs/$r/; done
steps() { ls runs/$1/rl3_policy_[0-9]*.npz | sed 's/.*_//; s/.npz//' | sort -n; }
ok() { for s in $(steps $1); do [ -f runs/$1/rl2_sl_$s.npz ] && [ $s -gt $FORK ] && echo $s; done; }
near() { awk -v t=$2 '{d = $1 - t; if (d < 0) d = -d; print d, $1}' <<< "$(ok $1)" | sort -n | head -1 | cut -d" " -f2; }
for r in $TR $CT; do  # a stalled learner makes "matched steps" silently compare old checkpoints (2026-09-27: OOM-killed learner)
  f=runs/$r/rl3_policy_$(ok $r | tail -1).npz; age=$(( $(date +%s) - $(stat -c %Y $f) ))
  [ $age -gt 3600 ] && echo "WARNING: newest checkpoint of $r is $((age / 60)) min old (learner stalled?)"
done
# matched steps: the run that is behind sets the two target steps, the other run takes its nearest checkpoints
if [ "$(ok $TR | tail -1)" -le "$(ok $CT | tail -1)" ]; then
  T=($(ok $TR | tail -2)); C=($(for t in "${T[@]}"; do near $CT $t; done | sort -un))
else
  C=($(ok $CT | tail -2)); T=($(for c in "${C[@]}"; do near $TR $c; done | sort -un))
fi
[ ${#T[@]} -ge 1 ] && [ ${#C[@]} -ge 1 ] || { echo "no checkpoints"; exit 1; }
cfg=(); for s in "${C[@]:1}"; do cfg+=("policy_path=runs/$CT/rl3_policy_$s.npz sl_path=runs/$CT/rl2_sl_$s.npz"); done
for s in "${T[@]}"; do cfg+=("policy_path=runs/$TR/rl3_policy_$s.npz sl_path=runs/$TR/rl2_sl_$s.npz"); done
echo "$(date +%T) $label: $CT ${C[*]} vs $TR ${T[*]} (fork $FORK), seeds $s0-$((s0 + n - 1))"
# no GNC kwargs on the command line: every policy flies with the tunables stored in its own npz (bwind 2 vs 4 etc.)
AGENT=rl3 .venv/bin/python dev/sweep_gnc.py --scenario 4 --seeds $s0-$((s0 + n - 1)) \
  --base policy_path=runs/$CT/rl3_policy_${C[0]}.npz sl_path=runs/$CT/rl2_sl_${C[0]}.npz --cfg "" "${cfg[@]}"

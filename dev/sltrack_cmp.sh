#!/usr/bin/env bash
# s4sltrack (SL tracks the policy) vs s4cont (fixed SL) at matched steps since the fork (DECISION_LOG §93 protocol):
# the two newest s4sltrack checkpoints and the two s4cont checkpoints nearest to them, each with its own saved SL,
# paired on fresh seeds.  dev/sltrack_cmp.sh <label> <first seed> [n seeds]   (local machine)
cd "$(dirname "$0")/.."
label=$1; s0=$2; n=${3:-384}; R='user@login:~/bpc'
rsync -az --include='rl3_policy_[0-9]*.npz' --include='rl2_sl_[0-9]*.npz' --exclude='*' $R/runs/s4sltrack/ runs/s4sltrack/
rsync -az --include='rl3_policy_[0-9]*.npz' --include='rl2_sl_[0-9]*.npz' --exclude='*' $R/runs/s4cont/ runs/s4cont/
steps() { ls runs/$1/rl3_policy_[0-9]*.npz | sed 's/.*_//; s/.npz//' | sort -n; }
T=($(for s in $(steps s4sltrack); do [ -f runs/s4sltrack/rl2_sl_$s.npz ] && [ $s -gt 7315200 ] && echo $s; done | tail -2))
C=($(for t in "${T[@]}"; do steps s4cont | awk -v t=$t '{d = $1 - t; if (d < 0) d = -d; print d, $1}' | sort -n | head -1 | cut -d" " -f2; done | sort -un))
[ ${#T[@]} -ge 1 ] && [ ${#C[@]} -ge 1 ] || { echo "no checkpoints"; exit 1; }
cfg=(); for s in "${C[@]:1}"; do cfg+=("policy_path=runs/s4cont/rl3_policy_$s.npz sl_path=runs/s4cont/rl2_sl_$s.npz"); done
for s in "${T[@]}"; do cfg+=("policy_path=runs/s4sltrack/rl3_policy_$s.npz sl_path=runs/s4sltrack/rl2_sl_$s.npz"); done
echo "$(date +%T) $label: s4cont ${C[*]} vs s4sltrack ${T[*]} (fork 7315200), seeds $s0-$((s0 + n - 1))"
AGENT=rl3 .venv/bin/python dev/sweep_gnc.py --scenario 4 --seeds $s0-$((s0 + n - 1)) \
  --base policy_path=runs/s4cont/rl3_policy_${C[0]}.npz sl_path=runs/s4cont/rl2_sl_${C[0]}.npz lag_comp=1 bwind=2 margin=0 \
  --cfg "" "${cfg[@]}"

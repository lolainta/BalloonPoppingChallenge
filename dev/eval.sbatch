#!/bin/bash
# Paired eval of agent variants on the cluster: task = variant x chunk of 8 seeds.
#   sbatch --array=0-59%38 dev/eval.sbatch     (5 variants x 12 chunks = seeds SEED0..SEED0+95)
#SBATCH -p devq -J bpc-eval -c 8 --mem-per-cpu=1G --time=30 --nice=1000
#SBATCH -o logs/eval_%A_%a.log  # per-job names: a new sweep must not overwrite the last one
VARIANTS=(
  "leg_dt=0.1"
  "leg_dt=0.1 ground_alt=30"
  "leg_dt=0.1 excess_dir=1"
  "leg_dt=0.1 excess_dir=2"
  "leg_dt=0.1 beam_N=3"
)
[ -n "$VARFILE" ] && mapfile -t VARIANTS < "$VARFILE"  # one kwargs line per variant
cd ~/bpc
N=${NCHUNK:-12}  # chunks of 8 seeds per variant
v=$((SLURM_ARRAY_TASK_ID / N)); c=$((SLURM_ARRAY_TASK_ID % N))
lo=$(( ${SEED0:-10000} + c * 8 )); hi=$((lo + 7))
echo "variant $v: ${VARIANTS[$v]}"
echo "agent ${AGENT:-beam}"
AGENT=${AGENT:-beam} ~/.local/bin/uv run python dev/run_seeds.py --seeds $lo-$hi --kw ${VARIANTS[$v]}

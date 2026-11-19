#!/bin/bash
# Leg-label collection: 250 tasks x 40 seeds, 8 cores each, <=20 concurrent (160 cores).
# ponytail: %20 cap + short tasks + nice keep other users' jobs flowing; raise %N if the queue is empty.
#SBATCH -p devq -J bpc-legs --array=0-249%20
#SBATCH -c 8 --mem-per-cpu=1G --time=25 --nice=1000
#SBATCH -o logs/legs_%a.log
cd ~/bpc
lo=$((SLURM_ARRAY_TASK_ID * 40)); hi=$((lo + 39))
~/.local/bin/uv run python dev/collect.py --seeds $lo-$hi --procs 8 --eps ${EPS:-0.3} --kw ${KW:-learned_legs=0} --out ${OUT:-data/legs}_$SLURM_ARRAY_TASK_ID.npz

#!/bin/bash
# BC demo collection: task = 40 seeds from SEED0, teacher = best Beam variant.
#SBATCH -p devq -J bpc-bc -c 8 --mem-per-cpu=2G --time=60 --nice=1000
#SBATCH -o logs/bc_%a.log
cd ~/bpc && mkdir -p data/bc
lo=$(( ${SEED0:-30000} + SLURM_ARRAY_TASK_ID * 40 )); hi=$((lo + 39))
AGENT=beam .venv/bin/python dev/bc_collect.py --seeds $lo-$hi --procs 8 --out data/bc/bc_$SLURM_ARRAY_TASK_ID.npz --kw leg_dt=0.1 keep_bias=100 replan_every=1

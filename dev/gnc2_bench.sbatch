#!/bin/bash
# GNC v2 flight benchmark on the cluster (dev/gnc2_bench.py): array task i = line i of the variants file VF
# (one 'name:k=v,...' per line), checkpoint CK, scenario SC -> logs/gnc2_<TAG>_<i>.log
#   sbatch --qos=devq_low --array=0-9 --export=ALL,CK=runs/s4bw4/rl3_policy_25478400.npz,VF=runs/gnc2_variants.txt,SC=1,TAG=t dev/gnc2_bench.sbatch
#SBATCH -p devq -J bpc-gnc2 -c 12 --mem=20G --time=1:00:00
#SBATCH -o logs/gnc2_%A_%a.out
cd ~/bpc
v=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" $VF); [ -n "$v" ] || exit 0
.venv/bin/python dev/gnc2_bench.py --scenario ${SC:-1} $CK "$v" 2>/dev/null > logs/gnc2_${TAG}_${SLURM_ARRAY_TASK_ID}.log

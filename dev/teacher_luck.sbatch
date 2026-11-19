#!/bin/bash
# dev/teacher_luck.py on SEEDS, one process per seed (DECISION_LOG §206). POLICY: the teacher's checkpoint.
#   sbatch -q normal -w cp13 -c 16 --mem=52800M --export=ALL,POLICY=runs/s4teach/rl3_policy_280225306.npz,SEEDS="90100 ..." dev/teacher_luck.sbatch
#SBATCH -p devq -J bpc-luck --time=5:00:00
#SBATCH -o logs/luck_%j.log
cd ~/bpc
echo $SEEDS | tr ' ' '\n' | xargs -P $SLURM_CPUS_PER_TASK -I{} ${PY:-/tmp/llt_py/.venv/bin/python} dev/teacher_luck.py $POLICY {}

#!/bin/bash
# Hybrid E2E M2 (DECISION_LOG §249 / §253 / §256): imitation training on a GPU (/tmp/llt_gpu; BC_PY=/tmp/llt_py/.venv/bin/python:
# the CPU venv, the job's cores), then the imitation net flies
# fresh scenario-40 seeds with its mean actions (/tmp/llt_py, CPU), launch hours cycling 0~21.
#   sbatch -q normal -w cp12 -c 16 --mem=48G --gres=gpu:1 --export=ALL,DEMOS=runs/e2e_bc/demos,OUT=runs/e2e_bc/bc_net.pt,EVAL=98500-98547 dev/e2e_bc.sbatch
#SBATCH -p devq -J bpc-e2ebc --time=4:00:00
#SBATCH -o logs/e2e_bc_%j.log
cd ~/bpc
${BC_PY:-/tmp/llt_gpu/.venv-gpu/bin/python} dev/e2e.py bc ${DEMOS:-runs/e2e_bc/demos} ${OUT:-runs/e2e_bc/bc_net.pt} ${EPOCHS:-40} || exit 1
E2E_KW="${E2E_KW:-policy_path=runs/s4ne6/rl3_policy_381960016.npz sl_path=runs/s4ne6/rl2_sl_381960016.npz att_fix=0.07}" PROCS=$SLURM_CPUS_PER_TASK \
  /tmp/llt_py/.venv/bin/python dev/e2e.py evalnet ${OUT:-runs/e2e_bc/bc_net.pt} ${EVAL%-*} ${EVAL#*-}

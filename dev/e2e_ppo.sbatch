#!/bin/bash
# Hybrid E2E M3 (DECISION_LOG §262): MODE=learn (GPU venv) or MODE=work (N episode processes, default all the job's cores) for RUN.
#   sbatch -q normal -w cp12 -c 4 --mem=16G --gres=gpu:1 --export=ALL,MODE=learn,RUN=runs/e2e_res1 dev/e2e_ppo.sbatch
#   sbatch -q normal -w cp13 -c 16 --mem=24G --export=ALL,MODE=work,RUN=runs/e2e_res1 dev/e2e_ppo.sbatch
#SBATCH -p devq -J bpc-e2eppo --time=40:00:00
#SBATCH -o logs/e2e_ppo_%j.log
cd ~/bpc
if [ "$MODE" = learn ]; then exec ${PY_GPU:-/tmp/llt_gpu/.venv-gpu/bin/python} dev/e2e_ppo.py learn $RUN; fi
E2E_KW="${E2E_KW:-policy_path=runs/s40a/rl3_policy_391874515.npz sl_path=runs/s40a/rl2_sl_391874515.npz att_fix=0.07}" \
  exec /tmp/llt_py/.venv/bin/python dev/e2e_ppo.py work $RUN ${N:-$SLURM_CPUS_PER_TASK}

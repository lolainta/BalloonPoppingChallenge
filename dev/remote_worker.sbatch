#!/bin/bash
# Env worker for multi-node PPO: RUN=<run dir> N=<envs> [LAUNCH=1]; learner uses --remote $RUN/workers
#SBATCH -p devq -J bpc-worker --mem=48G --time=40:00:00 --nice=1000
#SBATCH -o logs/worker_%j.log
cd ~/bpc
.venv/bin/python dev/remote_envs.py serve --run $RUN --n ${N:-15} ${LAUNCH:+--launch}

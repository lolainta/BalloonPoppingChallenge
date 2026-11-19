#!/bin/bash
# PPO guidance training on one 96-core node (SubprocVecEnv is single-node).
#   sbatch dev/ppo.sbatch            OUT/TIMESTEPS/ARGS via env
#SBATCH -p devq -J bpc-ppo -N1 -c 96 --mem=96G --time=16:00:00 --nice=1000
#SBATCH -o logs/ppo_%j.log
cd ~/bpc
${TRAINER:-.venv/bin/python dev/train_ppo.py} --envs ${ENVS:-90} --timesteps ${TIMESTEPS:-4000000} --out ${OUT:-runs/ppo1} ${ARGS:-}

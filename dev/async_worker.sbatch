#!/bin/bash
# Async PPO env workers on an extra node: RUN, N, [LAUNCH=1], SL env vars via --export
#SBATCH -p devq -J bpc-aworker --time=40:00:00 --nice=1000
#SBATCH -o logs/aworker_%j.log
cd ~/bpc
. dev/node_boot.sh  # §285
${PY:-.venv/bin/python} dev/async_ppo.py worker --run $RUN --n ${N:-15} ${LAUNCH:+--launch}

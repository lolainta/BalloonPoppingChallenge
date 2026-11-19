#!/bin/bash
# Async PPO learner + local workers on one GPU node: RUN, INIT | INIT_NPZ, LOCAL, [LAUNCH=1], [LR], [CKPT_EVERY], SL env vars via --export
#SBATCH -p devq -J bpc-async -N1 -c 96 --mem=160G --gres=gpu:1 --time=40:00:00 --nice=1000
#SBATCH -o logs/async_%j.log
# PY / SCRIPT: a node-local copy of the venv / code (slow NFS, DECISION_LOG §167); its rocketpy via PYTHONPATH in --export
cd ~/bpc
. dev/node_boot.sh  # §285: unpinned -> unpack the node-local copies if this node lacks them
${PY:-.venv-gpu/bin/python} ${SCRIPT:-dev/async_ppo.py} learn --run $RUN ${INIT:+--init $INIT} ${INIT_NPZ:+--init_npz $INIT_NPZ} --local ${LOCAL:-92} ${LAUNCH:+--launch} ${BATCH:+--batch $BATCH} ${LR:+--lr $LR} ${CKPT_EVERY:+--ckpt_every $CKPT_EVERY}

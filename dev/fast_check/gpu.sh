#!/usr/bin/env bash
# dev/fast_check/gpu.sh script.py args...: nice'd .venv-gpu python (local GPU 0) with env.sh's settings
source "$(dirname "$0")/env.sh"; cd $W
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-0} nice -n 10 ./.venv-gpu/bin/python "$@" 2>&1 | grep -v Warn

#!/bin/bash
# one <10 min resumable chunk, 12 flying slots total. usage: chunk.sh [extra args, e.g. --tsw 1.2 --tag appr12,chaos]
cd ./
.venv/bin/python dev/gnc2_seq/approach/approach_fork.py --mode both --seeds 46000-46047 --slots 12 --coords 2 \
  --deadline 540 --est 480 "$@" >> dev/gnc2_seq/approach/out/log_both.txt 2>&1
tail -n 8 dev/gnc2_seq/approach/out/log_both.txt; uptime

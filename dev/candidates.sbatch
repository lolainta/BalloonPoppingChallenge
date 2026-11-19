#!/bin/bash
# Keep runs/leaderboard_candidates.csv fresh: collect, secure weights, verify top results every 5 min.
#SBATCH -p devq -J bpc-cand -c 8 --mem=16G --time=40:00:00 --nice=1000
#SBATCH -o logs/candidates_%j.log
cd ~/bpc && .venv/bin/python dev/candidates.py --loop 300 --top 20 --procs 8

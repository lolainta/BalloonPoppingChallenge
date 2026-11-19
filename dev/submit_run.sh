#!/usr/bin/env bash
# Official leaderboard run on a chosen seed: sets scenario.random_seed in scenario_1_parameters.yaml, runs
# evaluate.py with bpc_submit_cfg.yaml, and ALWAYS restores the yaml (trap), even on error / Ctrl-C.
#   dev/submit_run.sh SEED [CFG]
set -euo pipefail
seed=$1; cfg=${2:-BalloonPoppingGymEnv/evaluation/configs/bpc_submit_cfg.yaml}
cd "$(dirname "$0")/.."
Y=BalloonPoppingGymEnv/envs/scenario_parameters/scenario_1_parameters.yaml
grep -q "CHANGE_ME" "$cfg" && { echo "fill in team_name / team_secret in $cfg first"; exit 1; }
cp "$Y" /tmp/scenario1_before_submit.yaml
trap 'cp /tmp/scenario1_before_submit.yaml "$Y"; echo "restored $Y (random_seed: $(grep -o "random_seed: [0-9a-z]*" "$Y"))"' EXIT
sed -i "s/^  random_seed: [0-9a-z]* /  random_seed: $seed /" "$Y"
grep -q "random_seed: $seed " "$Y" || { echo "could not set the seed"; exit 1; }
.venv/bin/python BalloonPoppingGymEnv/evaluation/evaluate.py "$cfg" 2>&1 | grep -viE "warn|Iteration|Completed 100|Total wall"
echo "submission file (upload this; it contains your team secret):"
ls -t BalloonPoppingGymEnv/evaluation/results/*.json | grep -v _trajectory | head -1

#!/usr/bin/env bash
# Qualification evaluation (10/10 19:00 GMT+8, DECISION_LOG §181): dev/qual_run.sh SEED CFG
#   CFG: the team's eval config (agent_module_path BalloonPoppingGymEnv/agents/bpc_s4_qual_160219460.py, agent_class_name RL3Agent,
#   agent_kwargs {}, scenario_number 4, render_mode null, team_name / team_secret / leaderboard_submission as the organisers say);
#   this script never reads it, it only hands the path to the official evaluator.
# 1) is SEED in our used-seed records? (USED -> tell the organisers, they redraw)  2) random_seed in the official scenario-4 yaml
# 3) the official evaluate.py  4) wall time + the newest result json (it holds team_secret: upload it, do not share it elsewhere)
set -euo pipefail
cd "$(dirname "$0")/.."
SEED=$1; CFG=$2
Y=BalloonPoppingGymEnv/envs/scenario_parameters/scenario_4_parameters.yaml
[[ $SEED =~ ^[0-9]+$ ]] || { echo "seed must be a number"; exit 1; }
[ -f "$CFG" ] || { echo "no config $CFG"; exit 1; }
echo "== 1. seed lookup"; .venv/bin/python dev/seed_check.py "$SEED" | tail -1
read -r -p "continue with seed $SEED? [y/N] " ok; [ "$ok" = y ] || exit 1
echo "== 2. $Y: random_seed -> $SEED"
sed -i -E "0,/^  random_seed:/s/^(  random_seed:)[^#]*/\1 $SEED /" "$Y"; grep -n "^  random_seed:" "$Y"
echo "== 3. official evaluator ($(date +%T))"; t0=$(date +%s)
.venv/bin/python BalloonPoppingGymEnv/evaluation/evaluate.py "$CFG"
echo "== 4. wall $(( $(date +%s) - t0 )) s (limit 600 s per episode); newest result:"
ls -t BalloonPoppingGymEnv/evaluation/results/*.json 2>/dev/null | head -1
.venv/bin/python -c "import sys; sys.path.insert(0, '.'); from dev.seed_registry import record; record([$SEED], 'qual_run', note='$CFG', scenario=4)"
echo "seed $SEED recorded in data/seed_registry/used_seeds.csv"

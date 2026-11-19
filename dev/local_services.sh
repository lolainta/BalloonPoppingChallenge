#!/usr/bin/env bash
# Mirror for the local dashboard: pulls checkpoints, eval results and the cluster-run comparison logs every 2 minutes.
# No compute here (user 2026-11-03: BPC compute runs on the cluster, not on the local machine): the held-out eval of
# each live learner's newest checkpoint moved to dev/det_eval_cluster.sh on login; the leaderboard-candidate pass
# (dev/candidates.py, off since §192) is gone.
#   setsid nohup dev/local_services.sh >> logs/local_services.log 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
R='user@login:~/bpc'  # quoted: ~ must expand on the cluster
while true; do
  rsync -az --prune-empty-dirs --include='*/' --include='det_eval*' --include='rl[23]_policy_[0-9]*.npz' --include='rl2_sl_[0-9]*.npz' --include='tune*/progress.csv' --include='tune*/status.json' --include='tune*/seeds.txt' \
    --exclude='*' $R/runs/ runs/
  rsync -az --include='eval_*_*.log' --include='*_s40*_*.log' --include='*_s41*_*.log' --exclude='*' $R/logs/ logs/  # + the cluster-run comparisons (§252)
  sleep 120
done

#!/usr/bin/env bash
#SBATCH -J bpc-posthoc
#SBATCH -q normal
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 0:45:00
#SBATCH --exclude=cp08,cp11,cp14,cp18,cp20
#SBATCH -o logs/posthoc_finale_%j.log
# DECISION_LOG §521 (user 11/8, after the official results and our upload): post-hoc only, never uploaded. The earlier
# finale files and the last training exports on the two finale seeds at hour 15, all on one node. One private copy per seed
# (finale_eval.py edits that copy's yaml), its default cfg (leaderboard_submission false), seeds recorded in the shared
# registry. Extra agent files come from ~/bpc_posthoc/agents.
set -u
cd ~/bpc
X=$HOME/bpc_posthoc/agents
AG="bpc_s40_fin_pqp.py bpc_s40_fin_phe.py bpc_s40_fb_cur.py bpc_s40_dry_navavg.py bpc_s40_dry_soup3w64t2.py bpc_s40_ph_s40l.py bpc_s40_ph_s40m.py bpc_s40_ph_s40n.py"
for s in 1353635681 1715955564; do
  (
    P=$(mktemp -d /tmp/bpc_ph_${s}_XXXX)
    R=/BalloonPoppingGymEnv/evaluation/results
    rsync -a --include="$R/*.py" --exclude="$R/*" --exclude=bpc_submit_cfg.yaml --exclude=qual_cfg_test.yaml \
      --exclude=__pycache__ BalloonPoppingGymEnv $P/
    mkdir -p $P/dev/qual_check $P/logs
    cp dev/finale_eval.py dev/seed_registry.py $P/dev/ && cp dev/qual_check/final_cfg.yaml $P/dev/qual_check/ \
      && cp $X/*.py $P/BalloonPoppingGymEnv/agents/
    cd $P
    for f in $AG; do
      SEED_REGISTRY=$HOME/bpc/data/seed_registry/used_seeds.csv ~/bpc/.venv/bin/python dev/finale_eval.py $s 15 \
        --agent BalloonPoppingGymEnv/agents/$f --force > logs/run.out 2>&1
      cat logs/run.out > ~/bpc/logs/posthoc_${s}_${f%.py}.log
    done
    cp logs/finale_eval.tsv ~/bpc/logs/posthoc_finale_$s.tsv
    cd / && rm -rf $P
  ) &
done
wait
echo "host $(hostname)"
cut -f2-9 logs/posthoc_finale_1353635681.tsv logs/posthoc_finale_1715955564.tsv | grep -v '^seed'

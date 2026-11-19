#!/usr/bin/env bash
# DECISION_LOG §508 (user 11/8 00:15: "跑確認，收訓練"): confirmation of the probability-averaging ensemble after the §507
# screening (+0.042 +- 0.033, failed z >= 2.24; the user chose to confirm anyway). Phase E layout: 8 sets pca-h x 8 hours
# x 48 seeds = 3,072 fresh seeds 10067000-10070071, one 8-core job per set and hour, the training shut down first.
# - Base = phe pair files + nav_avg=1e-4 (the finale file's weights). Members = the three branch averages phe is the mean
#   of (runs/oracle/rl3_policy_phe{l,m,n}.npz, SL v4w), as in the screening (dev/launch_pq.sh).
#   cfg1 pqp = ens_agg=prob (the candidate);  cfg2 pql = logit ensemble, a control only (rejected in §429 / §448 / §507).
# - Pre-registered rule, unchanged: P=pc S0=10067000 W1=pqp W2=pql ONLY1=1 (§509: solo) BN=phe BF=bpc_s40_fin_phe.py bash dev/pool_phc.sh
#   -> cfg1 replaces phe only with diff >= +0.04, z >= 2.24, no hour below -0.15 and the higher mean of the two; then the
#   single-file packing with ensemble support + the official 16/16 on 10070100-10070115 before 11/8 11:30. Else phe.
# §509: pooled with ONLY1=solo instead (pqp judged by its own three conditions; pql never blocks it).
set -euo pipefail
cd ~/bpc
O=runs/oracle; S0=10067000; NS=3072
[ ! -e runs/.cmp_parts/probe_pca ] || { echo "ABORT: probe_pca exists"; exit 1; }
n=$(cut -d, -f1 data/seed_registry/used_seeds.csv | grep -E '^[0-9]+$' | awk -v a=$S0 -v b=$((S0 + NS + 116)) '$1 >= a && $1 < b' | wc -l)
[ "$n" -eq 0 ] || { echo "ABORT: seed registry overlaps [$S0, $((S0 + NS + 116)))"; exit 1; }
md5sum $O/rl3_policy_phe.npz $O/rl3_policy_phe{l,m,n}.npz $O/rl2_sl_phe{,l,m,n}.npz \
  BalloonPoppingGymEnv/agents/{gnc_agent,rl2_agent,dp_plan,rl3_agent,rl2_sl}.py | tee $O/pc.md5
C1="policy_path=$O/rl3_policy_phel.npz sl_path=$O/rl2_sl_phel.npz ens=$O/rl3_policy_phem.npz,$O/rl3_policy_phen.npz"
CP="$C1 ens_agg=prob"
for k in 0 1 2 3 4 5 6 7; do
  AGENT_CLS=rl3 TAG=pc$(echo abcdefgh | cut -c$((k + 1))) HOURS="0 3 6 9 12 15 18 21" N=48 S0=$((S0 + 384 * k)) CORES=8 MEM=8G \
    TIME=1:30:00 EXCLUDE=cp11,cp14,cp18,cp20 BASE=$O/rl3_policy_phe.npz BASE_EXTRA=nav_avg=1e-4 CFGS="$CP;$C1" \
    setsid nohup bash dev/probe_cmp.sh >> logs/probe_sched.log 2>&1 < /dev/null &
done
sleep 30
squeue -u $USER -h -n bpc-cmp -o "%T" | sort | uniq -c || true

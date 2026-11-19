#!/usr/bin/env bash
# DECISION_LOG §505 (user 11/7 23:00: probability averaging over the three branch soups, untried): screening probe, phe's
# layout halved: 2 sets pqa-b x 8 hours x 48 seeds = 768 seeds 10066200-10066967, one 8-core job per set and hour.
# - Base = phe pair files + nav_avg=1e-4 (the finale file's weights). Members = the three branch averages phe was built
#   from (runs/<r>/avg105_<END> copied to runs/oracle/rl3_policy_phe{l,m,n}.npz, SL v4w): phe = their mean exactly.
#   cfg1 pql  = logit ensemble (ens=, the §429 / §448 kind: mean logits);  cfg2 pqp = ens_agg=prob (mean probabilities).
# - Pre-registered: a config passes the screening with diff >= +0.04, z >= 2.24 and no hour below -0.15 on the 768 seeds
#   (pooled by dev/pool_pq.sh); only then a 3,072-seed confirmation in Phase F's layout on fresh seeds, same rule, plus
#   single-file ens support and the official 16/16 before 11/8 11:30. Otherwise the finale keeps phe.
set -euo pipefail
cd ~/bpc
O=runs/oracle; S0=10066200; NS=768
[ ! -e runs/.cmp_parts/probe_pqa ] || { echo "ABORT: probe_pqa exists"; exit 1; }
n=$(cut -d, -f1 data/seed_registry/used_seeds.csv | grep -E '^[0-9]+$' | awk -v a=$S0 -v b=$((S0 + NS)) '$1 >= a && $1 < b' | wc -l)
[ "$n" -eq 0 ] || { echo "ABORT: seed registry overlaps [$S0, $((S0 + NS)))"; exit 1; }
md5sum $O/rl3_policy_phe.npz $O/rl3_policy_phe{l,m,n}.npz $O/rl2_sl_phe{,l,m,n}.npz BalloonPoppingGymEnv/agents/rl3_agent.py | tee $O/pq.md5
C1="policy_path=$O/rl3_policy_phel.npz sl_path=$O/rl2_sl_phel.npz ens=$O/rl3_policy_phem.npz,$O/rl3_policy_phen.npz"
C2="$C1 ens_agg=prob"
for k in 0 1; do
  AGENT_CLS=rl3 TAG=pq$(echo ab | cut -c$((k + 1))) HOURS="0 3 6 9 12 15 18 21" N=48 S0=$((S0 + 384 * k)) CORES=8 MEM=8G \
    TIME=1:30:00 EXCLUDE=cp11,cp14,cp18,cp20 BASE=$O/rl3_policy_phe.npz BASE_EXTRA=nav_avg=1e-4 CFGS="$C1;$C2" \
    setsid nohup bash dev/probe_cmp.sh >> logs/probe_sched.log 2>&1 < /dev/null &
done
sleep 20
squeue -u $USER -h -n bpc-cmp -o "%T" | sort | uniq -c || true

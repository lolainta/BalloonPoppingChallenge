#!/usr/bin/env bash
# DECISION_LOG §480 Phase D (pre-registered 11/6 13:15; user 11/6: no freeze, iterating the SL is fine): the eval-time SL
# swap cur + v5 vs cur + v4w vs cur in Phase C's layout (dev/launch_phc.sh: 8 sets phda-h x 8 hours x 48 seeds, one 8-core
# 1-h job per set and hour through dev/probe_cmp.sh). Only after gate A'' (dev/sl5_fwd_eval.py, printed at the end of the
# newest logs/sl40_v5_<job>.log) named a candidate; its CANDIDATE line picks v5w or v5b.
# - Pair files in runs/oracle, so dev/qual_check/offeval_8h.sbatch's ${NPZ/rl3_policy/rl2_sl} finds the winner's SL:
#     base cur + v2 (rl3_policy_cur / rl2_sl_cur), cfg1 cur + v5 (rl3_policy_curv5 = cur / rl2_sl_curv5 = the pick),
#     cfg2 cur + v4w (rl3_policy_curv4 / rl2_sl_curv4, Phase C's files). RL3Agent, nav_avg=1e-4, argmax.
#   runs/oracle/phd.md5 records them and the agent sources; W=curv5 P=phd bash dev/pack_phc.sh checks them.
# - 3,072 seeds 10054000-10057071: set k, hour index i: 10054000 + 384 k + 48 i .. + 47.
# - Pool: P=phd S0=10054000 W1=curv5 ONLY1=1 bash dev/pool_phc.sh; Phase C's rule (the best config >= +0.04, z >= 2.24,
#   no hour below -0.15, else cur), and cfg2 is a control only (§481: Phase C already rejected cur + v4w, so a cfg2 win
#   never switches). Deadline 11/7 22:00 (§480).
#   bash dev/launch_phd.sh
set -euo pipefail
cd ~/bpc
O=runs/oracle; V4=49feb737c5b430c8c937ac0a506ead0b; V2=bb6459d7d66389ce977778e1fadac775; CUR=47f7b16d5d28d396b1364d2cfe01ce91
S0=10054000; NS=3072
md5() { md5sum < "$1" | cut -c1-32; }
log=$(ls -t logs/sl40_v5_*.log | head -1)
pick=$(grep -oE "^CANDIDATE: v5[wb]" $log | cut -d' ' -f2) || true
[ -n "$pick" ] || { echo "ABORT: no CANDIDATE in $log (gate A'' failed, or the job is still running) -> v4w stays, no Phase D"; exit 1; }
V5F=runs/sl40/rl2_sl_s40lmn_$pick.npz; [ -s $V5F ] || { echo "ABORT: $V5F missing"; exit 1; }
[ ! -e $O/rl2_sl_curv5.npz ] && [ ! -e runs/.cmp_parts/probe_phda ] || { echo "ABORT: Phase D already launched"; exit 1; }
n=$(cut -d, -f1 data/seed_registry/used_seeds.csv | grep -E '^[0-9]+$' | awk -v a=$S0 -v b=$((S0 + NS)) '$1 >= a && $1 < b' | wc -l)
[ "$n" -eq 0 ] || { echo "ABORT: seed registry overlaps [$S0, $((S0 + NS)))"; exit 1; }
[ "$(md5 $O/rl3_policy_cur.npz)" = $CUR ] && [ "$(md5 $O/rl2_sl_cur.npz)" = $V2 ] && [ "$(md5 $O/rl3_policy_curv4.npz)" = $CUR ] \
  && [ "$(md5 $O/rl2_sl_curv4.npz)" = $V4 ] || { echo "ABORT: cur / v2 / curv4 pair md5"; exit 1; }
cp $O/rl3_policy_cur.npz $O/rl3_policy_curv5.npz; cp $V5F $O/rl2_sl_curv5.npz
undo() { rm -f $O/rl3_policy_curv5.npz $O/rl2_sl_curv5.npz; echo "ABORT: $1 (pair files removed, rerun after the fix)"; exit 1; }
[ "$(md5 $O/rl2_sl_curv5.npz)" = "$(md5 $V5F)" ] && [ "$(md5 $O/rl3_policy_curv5.npz)" = $CUR ] || undo "pair file md5"
.venv/bin/python -c "
import numpy as np
from BalloonPoppingGymEnv.agents.rl2_sl import load_sl
t, p = load_sl('$O/rl2_sl_curv5.npz')(np.zeros((3, 15)), np.zeros(6))
assert t.shape == p.shape == (3,) and np.isfinite(t).all() and np.isfinite(p).all(); print('v5 pair file loads: $pick')" \
  || undo "the v5 pair file does not load"
echo "Phase D: base cur + v2, cfg1 cur + $pick, cfg2 cur + v4w; seeds $S0-$((S0 + NS - 1))"
md5sum $O/rl3_policy_curv5.npz $O/rl2_sl_curv5.npz $O/rl3_policy_curv4.npz $O/rl2_sl_curv4.npz \
  BalloonPoppingGymEnv/agents/{gnc_agent,rl2_agent,dp_plan,rl3_agent,rl2_sl}.py | tee $O/phd.md5
md5sum dev/sweep_gnc.py dev/probe_cmp.sh dev/line_cmp.sbatch
C1="policy_path=$O/rl3_policy_curv5.npz sl_path=$O/rl2_sl_curv5.npz"
C2="policy_path=$O/rl3_policy_curv4.npz sl_path=$O/rl2_sl_curv4.npz"
for k in 0 1 2 3 4 5 6 7; do
  AGENT_CLS=rl3 TAG=phd$(echo abcdefgh | cut -c$((k + 1))) HOURS="0 3 6 9 12 15 18 21" N=48 S0=$((S0 + 384 * k)) CORES=8 MEM=8G \
    TIME=1:00:00 EXCLUDE=cp11,cp14,cp18,cp20 BASE=$O/rl3_policy_cur.npz BASE_EXTRA=nav_avg=1e-4 CFGS="$C1;$C2" \
    setsid nohup bash dev/probe_cmp.sh >> logs/probe_sched.log 2>&1 < /dev/null &
done
sleep 30
squeue -u $USER -h -n bpc-cmp -o "%T" | sort | uniq -c || true

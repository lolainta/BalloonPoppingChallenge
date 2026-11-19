#!/usr/bin/env bash
# DECISION_LOG §496 Phase E (pre-registered 11/7 13:15; user 11/6: keep training until the finale): the post-Phase-C soup
# vs cur, in Phase C's layout (dev/launch_phc.sh: 8 sets phea-h x 8 hours x 48 seeds, one 8-core 1-h job per set and hour
# through dev/probe_cmp.sh). Why now: cur scores 8.708 on the 96 det_eval seeds (logs/det_eval_cur.log) and the newest
# single s40l / s40n exports score 8.70-8.77 there, i.e. the singles have caught up with cur's weights (§496).
# - Per branch: every export after the Phase C soup end (END below) must carry the v4w SL copy (md5 49feb737), else ABORT;
#   an export younger than 2 min is left out (async_ppo.py writes its SL copy after the policy). n_e = those exports;
#   a branch with n_e < 48 -> ABORT. M = min(128, min n_e): per branch the average of its newest M exports, then the
#   equal-weight mean (dev/soup_avg.py; 0-d / non-float keys from s40l). Nothing in the window was in Phase C's soup.
# - Configs on the same seeds, each with an explicit policy_path + sl_path (pair files in runs/oracle, so
#   dev/qual_check/offeval_8h.sbatch's ${NPZ/rl3_policy/rl2_sl} finds the winner's SL):
#     base cur + v2 (rl3_policy_cur / rl2_sl_cur), cfg1 l+m+n soup + v4w (rl3_policy_phe / rl2_sl_phe),
#     cfg2 l+n soup + v4w (rl3_policy_phe2 / rl2_sl_phe2; s40m's singles trail cur by 0.1-0.18 on the det_eval seeds).
#   RL3Agent, nav_avg=1e-4, argmax. runs/oracle/phe.md5 records the pair files and the agent sources.
# - 3,072 seeds 10058000-10061071: set k, hour index i: 10058000 + 384 k + 48 i .. + 47 (registry max 10057071).
# - Pool: P=phe S0=10058000 W1=phe W2=phe2 bash dev/pool_phc.sh; Phase C's rule unchanged: the better config replaces cur
#   only with >= +0.04, z >= 2.24 and no hour below -0.15, else cur. Pack: W=<phe|phe2> P=phe bash dev/pack_phc.sh
#   (bpc_s40_fin_<W>.py + official evaluate.py 16/16 on 10052100-10052115). Deadline: VERDICT and 16/16 by 11/7 21:00,
#   else the finale keeps cur. Only once (an existing soup or probe_phea dir aborts).
#   DRY=1 bash dev/launch_phe.sh          bash dev/launch_phe.sh
set -euo pipefail
cd ~/bpc
O=runs/oracle; V4F=runs/sl40/rl2_sl_s40lmn_v4w.npz
V4=49feb737c5b430c8c937ac0a506ead0b; V2=bb6459d7d66389ce977778e1fadac775; CUR=47f7b16d5d28d396b1364d2cfe01ce91
declare -A END=([s40l]=645092013 [s40m]=641833757 [s40n]=660914320)  # Phase C's soup windows ended here (§476)
S0=10058000; NS=3072; MINE=48
md5() { md5sum < "$1" | cut -c1-32; }
[ -n "${DRY:-}" ] || { [ ! -e $O/rl3_policy_phe.npz ] && [ ! -e runs/.cmp_parts/probe_phea ] || { echo "ABORT: Phase E already built or launched"; exit 1; }; }
n=$(cut -d, -f1 data/seed_registry/used_seeds.csv | grep -E '^[0-9]+$' | awk -v a=$S0 -v b=$((S0 + NS)) '$1 >= a && $1 < b' | wc -l)
[ "$n" -eq 0 ] || { echo "ABORT: seed registry overlaps [$S0, $((S0 + NS)))"; exit 1; }
[ "$(md5 $O/rl3_policy_cur.npz)" = $CUR ] && [ "$(md5 $O/rl2_sl_cur.npz)" = $V2 ] && [ "$(md5 $V4F)" = $V4 ] \
  || { echo "ABORT: cur / v2 / v4w md5"; exit 1; }
declare -A NE NEW
for r in s40l s40m s40n; do
  [ "$(md5 runs/$r/rl2_sl.npz)" = $V4 ] || { echo "ABORT: $r live SL is not v4w"; exit 1; }
  post=$(ls runs/$r/rl3_policy_[0-9]*.npz | sed 's/.*_//; s/\.npz$//' | sort -n | awk -v a=${END[$r]} '$1 > a')
  last=$(echo "$post" | tail -1)
  if [ $(($(date +%s) - $(stat -c %Y runs/$r/rl3_policy_$last.npz))) -lt 120 ]; then post=$(echo "$post" | sed '$d'); fi
  for s in $post; do
    f=runs/$r/rl2_sl_$s.npz; [ -e $f ] || { echo "ABORT: $r export $s has no SL copy (learner stopped mid-export?)"; exit 1; }
    h=$(md5 $f); [ $h = $V4 ] || { echo "ABORT: $r export $s SL copy md5 $h is not v4w"; exit 1; }
  done
  NE[$r]=$(echo "$post" | wc -l); NEW[$r]=$(echo "$post" | tail -1)
  echo "$r: ${NE[$r]} exports after Phase C's window end ${END[$r]}, all v4w; newest ${NEW[$r]}"
  [ ${NE[$r]} -ge $MINE ] || { echo "ABORT: $r has only ${NE[$r]} exports after Phase C (< $MINE)"; exit 1; }
done
M=128; for r in s40l s40m s40n; do if [ ${NE[$r]} -lt $M ]; then M=${NE[$r]}; fi; done
echo "soup phe: s40l s40m s40n, the newest $M exports of each; soup phe2: s40l s40n, the same windows"
[ -z "${DRY:-}" ] || { echo "DRY: stop before writing"; exit 0; }
P=.venv/bin/python
nice -n 19 $P dev/soup_avg.py $O/rl3_policy_phe.npz $M s40l:${NEW[s40l]} s40m:${NEW[s40m]} s40n:${NEW[s40n]}
nice -n 19 $P dev/soup_avg.py $O/rl3_policy_phe2.npz $M s40l:${NEW[s40l]} s40n:${NEW[s40n]}
cp $V4F $O/rl2_sl_phe.npz; cp $V4F $O/rl2_sl_phe2.npz
[ "$(md5 $O/rl2_sl_phe.npz)" = $V4 ] && [ "$(md5 $O/rl2_sl_phe2.npz)" = $V4 ] || { echo "ABORT: pair file md5"; exit 1; }
md5sum $O/rl3_policy_phe.npz $O/rl2_sl_phe.npz $O/rl3_policy_phe2.npz $O/rl2_sl_phe2.npz \
  BalloonPoppingGymEnv/agents/{gnc_agent,rl2_agent,dp_plan,rl3_agent,rl2_sl}.py | tee $O/phe.md5
md5sum dev/sweep_gnc.py dev/probe_cmp.sh dev/line_cmp.sbatch
$P - <<'PY'
import numpy as np
L = lambda n: dict(np.load(f"runs/oracle/rl3_policy_{n}.npz", allow_pickle=True))
cur = L("cur")
f = [k for k, v in cur.items() if v.dtype.kind == "f" and v.ndim >= 1]
for n in ("phe", "phe2"):
    s = L(n)
    assert sorted(s) == sorted(cur), set(s) ^ set(cur)
    assert all(s[k].shape == cur[k].shape and np.isfinite(s[k]).all() for k in f)
    print(n, "non-averaged keys that differ from cur:", {k: (s[k].tolist(), cur[k].tolist()) for k in cur if k not in f and not np.array_equal(s[k], cur[k])})
    nrm = np.sqrt(sum(((s[k].astype(float) - cur[k].astype(float)) ** 2).sum() for k in f) / sum((cur[k].astype(float) ** 2).sum() for k in f))
    print(f"|{n} - cur| / |cur| = {nrm:.4f}")
PY
C1="policy_path=$O/rl3_policy_phe.npz sl_path=$O/rl2_sl_phe.npz"
C2="policy_path=$O/rl3_policy_phe2.npz sl_path=$O/rl2_sl_phe2.npz"
for k in 0 1 2 3 4 5 6 7; do
  AGENT_CLS=rl3 TAG=phe$(echo abcdefgh | cut -c$((k + 1))) HOURS="0 3 6 9 12 15 18 21" N=48 S0=$((S0 + 384 * k)) CORES=8 MEM=8G \
    TIME=1:00:00 EXCLUDE=cp11,cp14,cp18,cp20 BASE=$O/rl3_policy_cur.npz BASE_EXTRA=nav_avg=1e-4 CFGS="$C1;$C2" \
    setsid nohup bash dev/probe_cmp.sh >> logs/probe_sched.log 2>&1 < /dev/null &
done
sleep 30
squeue -u $USER -h -n bpc-cmp -o "%T" | sort | uniq -c || true

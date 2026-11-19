#!/usr/bin/env bash
# DECISION_LOG §460 Phase C (pre-registered 11/5 01:15; base = cur by probe_sel2 rule (C), §462; packing and guards §463):
# the SL v4 soup vs cur.
# - Per branch: every export since the switch point (the first export with a v4 SL copy, §461) must carry the v4w copy
#   (md5 49feb737), else ABORT (a v2 copy or a non-v4 live SL means a revert: no Phase C). An export younger than 2 min is
#   left out (async_ppo.py writes its SL copy after the policy). n_b = the exports from switch + 2.0M on.
# - Branches with n_b < 48 are left out; fewer than 2 left -> no Phase C. M = min(128, min n_b): per branch the average of
#   its newest M exports, then the equal-weight mean of the branches (dev/soup_avg.py; 0-d / non-float keys from s40l).
# - Configs on the same seeds, each with an explicit policy_path + sl_path. The SLs are dedicated runs/oracle pair files, so
#   dev/qual_check/offeval_8h.sbatch's ${NPZ/rl3_policy/rl2_sl} finds the right SL when the winner is packaged:
#     base cur + v2 (rl3_policy_cur / rl2_sl_cur), cfg1 v4 soup + v4 (rl3_policy_phc / rl2_sl_phc),
#     cfg2 cur + v4 (rl3_policy_curv4 = cur / rl2_sl_curv4). RL3Agent, nav_avg=1e-4, argmax.
#   runs/oracle/phc.md5 records the pair files and the agent sources; dev/pack_phc.sh checks them.
# - 3,072 seeds 10049000-10052071 = 8 sets (phca-h) x 8 hours x 48; set k, hour index i: 10049000 + 384 k + 48 i .. + 47.
#   One job per set and hour: 144 episodes, 8 cores, 8G, 1 h (cp19 ran 192 in 47 min, §463); not on cp11 (slow NFS),
#   cp14 / cp18 (workers), cp20 (other numerics, §452).
# - The real run only 11/6 05:00-12:00 (ANYTIME=1 overrides), only once (an existing soup or probe_phca dir aborts), and
#   only with G3_OK=1 after reading the +2.3M checks it prints. Pool with dev/pool_phc.sh.
# - DRY=1: the checks and the plan only, nothing written (DRY_BURN / DRY_MINB exercise the counting before the branches
#   reach +2.0M; ignored without DRY).
#   DRY=1 bash dev/launch_phc.sh          G3_OK=1 bash dev/launch_phc.sh
set -euo pipefail
cd ~/bpc
O=runs/oracle; V4F=runs/sl40/rl2_sl_s40lmn_v4w.npz
V4=49feb737c5b430c8c937ac0a506ead0b; V2=bb6459d7d66389ce977778e1fadac775; CUR=47f7b16d5d28d396b1364d2cfe01ce91
declare -A SW=([s40l]=638088634 [s40m]=634600062 [s40n]=649847571)
S0=10049000; NS=3072; BURN=2000000; MINB=48
if [ -n "${DRY:-}" ]; then BURN=${DRY_BURN:-$BURN}; MINB=${DRY_MINB:-$MINB}; else
  now=$(date +%s)
  [ -n "${ANYTIME:-}" ] || { [ $now -ge $(date -d '2026-11-06 05:00' +%s) ] && [ $now -le $(date -d '2026-11-06 12:00' +%s) ]; } \
    || { echo "ABORT: outside the 11/6 05:00-12:00 window (§460)"; exit 1; }
  [ ! -e $O/rl3_policy_phc.npz ] && [ ! -e runs/.cmp_parts/probe_phca ] || { echo "ABORT: Phase C already built or launched"; exit 1; }
  for r in s40l s40m s40n; do echo "== G3 $r"; tail -n 12 logs/sl4_${r}_2.3M.log 2>/dev/null || echo "(no result yet)"; done
  [ -n "${G3_OK:-}" ] || { echo "ABORT: read the G3 results above (§460: pooled < -0.15 -> revert, no Phase C), then G3_OK=1"; exit 1; }
fi
md5() { md5sum < "$1" | cut -c1-32; }
n=$(cut -d, -f1 data/seed_registry/used_seeds.csv | grep -E '^[0-9]+$' | awk -v a=$S0 -v b=$((S0 + NS)) '$1 >= a && $1 < b' | wc -l)
[ "$n" -eq 0 ] || { echo "ABORT: seed registry overlaps [$S0, $((S0 + NS)))"; exit 1; }
[ "$(md5 $O/rl3_policy_cur.npz)" = $CUR ] && [ "$(md5 $O/rl2_sl_cur.npz)" = $V2 ] && [ "$(md5 $V4F)" = $V4 ] \
  || { echo "ABORT: cur / v2 / v4w md5"; exit 1; }
declare -A NB NEW; keep=()
for r in s40l s40m s40n; do
  [ "$(md5 runs/$r/rl2_sl.npz)" = $V4 ] || { echo "ABORT: $r live SL is not v4w (reverted -> no Phase C, §460)"; exit 1; }
  post=$(ls runs/$r/rl3_policy_[0-9]*.npz | sed 's/.*_//; s/\.npz$//' | sort -n | awk -v a=${SW[$r]} '$1 >= a')
  [ "$(echo "$post" | head -1)" = ${SW[$r]} ] || { echo "ABORT: $r switch export ${SW[$r]} missing"; exit 1; }
  last=$(echo "$post" | tail -1)
  if [ $(($(date +%s) - $(stat -c %Y runs/$r/rl3_policy_$last.npz))) -lt 120 ]; then post=$(echo "$post" | sed '$d'); fi
  for s in $post; do
    f=runs/$r/rl2_sl_$s.npz; [ -e $f ] || { echo "ABORT: $r export $s has no SL copy (learner stopped mid-export?)"; exit 1; }
    h=$(md5 $f); [ $h = $V4 ] || { echo "ABORT: $r export $s SL copy md5 $h ($([ $h = $V2 ] && echo 'v2: a revert, no Phase C' || echo 'neither v4w nor v2'))"; exit 1; }
  done
  NB[$r]=$(echo "$post" | awk -v a=$((${SW[$r]} + BURN)) '$1 >= a' | wc -l); NEW[$r]=$(echo "$post" | tail -1)
  echo "$r: $(echo "$post" | wc -l) exports since the switch ${SW[$r]}, all v4w; n_b ${NB[$r]} (>= $((${SW[$r]} + BURN))), newest ${NEW[$r]}"
  if [ ${NB[$r]} -ge $MINB ]; then keep+=($r); fi
done
[ ${#keep[@]} -ge 2 ] || { echo "NO PHASE C: ${#keep[@]} branch(es) with n_b >= $MINB (§460)"; exit 1; }
M=128; for r in "${keep[@]}"; do if [ ${NB[$r]} -lt $M ]; then M=${NB[$r]}; fi; done
echo "soup: ${keep[*]}, the newest $M exports of each"
[ -z "${DRY:-}" ] || { echo "DRY: stop before writing"; exit 0; }
P=.venv/bin/python
nice -n 19 $P dev/soup_avg.py $O/rl3_policy_phc.npz $M $(for r in "${keep[@]}"; do echo $r:${NEW[$r]}; done)
cp $V4F $O/rl2_sl_phc.npz; cp $O/rl3_policy_cur.npz $O/rl3_policy_curv4.npz; cp $V4F $O/rl2_sl_curv4.npz
[ "$(md5 $O/rl2_sl_phc.npz)" = $V4 ] && [ "$(md5 $O/rl2_sl_curv4.npz)" = $V4 ] && [ "$(md5 $O/rl3_policy_curv4.npz)" = $CUR ] \
  || { echo "ABORT: pair file md5"; exit 1; }
md5sum $O/rl3_policy_phc.npz $O/rl2_sl_phc.npz $O/rl3_policy_curv4.npz $O/rl2_sl_curv4.npz \
  BalloonPoppingGymEnv/agents/{gnc_agent,rl2_agent,dp_plan,rl3_agent,rl2_sl}.py | tee $O/phc.md5
md5sum dev/sweep_gnc.py dev/probe_cmp.sh dev/line_cmp.sbatch
$P - <<'PY'
import numpy as np
L = lambda n: dict(np.load(f"runs/oracle/rl3_policy_{n}.npz", allow_pickle=True))
phc, cur = L("phc"), L("cur")
assert sorted(phc) == sorted(cur), set(phc) ^ set(cur)
f = [k for k, v in cur.items() if v.dtype.kind == "f" and v.ndim >= 1]
assert all(phc[k].shape == cur[k].shape and np.isfinite(phc[k]).all() for k in f)
print("non-averaged keys that differ from cur:", {k: (phc[k].tolist(), cur[k].tolist()) for k in cur if k not in f and not np.array_equal(phc[k], cur[k])})
nrm = np.sqrt(sum(((phc[k].astype(float) - cur[k].astype(float)) ** 2).sum() for k in f) / sum((cur[k].astype(float) ** 2).sum() for k in f))
print(f"|phc - cur| / |cur| = {nrm:.4f}")
PY
C1="policy_path=$O/rl3_policy_phc.npz sl_path=$O/rl2_sl_phc.npz"
C2="policy_path=$O/rl3_policy_curv4.npz sl_path=$O/rl2_sl_curv4.npz"
for k in 0 1 2 3 4 5 6 7; do
  AGENT_CLS=rl3 TAG=phc$(echo abcdefgh | cut -c$((k + 1))) HOURS="0 3 6 9 12 15 18 21" N=48 S0=$((S0 + 384 * k)) CORES=8 MEM=8G \
    TIME=1:00:00 EXCLUDE=cp11,cp14,cp18,cp20 BASE=$O/rl3_policy_cur.npz BASE_EXTRA=nav_avg=1e-4 CFGS="$C1;$C2" \
    setsid nohup bash dev/probe_cmp.sh >> logs/probe_sched.log 2>&1 < /dev/null &
done
sleep 30
squeue -u $USER -h -n bpc-cmp -o "%T" | sort | uniq -c || true

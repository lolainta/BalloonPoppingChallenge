#!/usr/bin/env bash
# DECISION_LOG §502 Phase F (pre-registered 11/7 21:00; user 11/7: try untried related work): one last re-soup vs the finale
# file's weights phe (bpc_s40_fin_phe.py), in Phase E's layout (8 sets phfa-h x 8 hours x 48 seeds, one 8-core 1-h job per
# set and hour through dev/probe_cmp.sh). Usually started by dev/sched_phf.sh at 01:45 11/8.
# - Per branch: every export after phe's soup window (END below) must carry the v4w SL copy, else ABORT; an export younger
#   than 2 min is left out; n_e < MINE (48) -> ABORT. M = min(128, min n_e).
#   cfg2 phf2 = equal-weight 3-branch soup of the newest M exports (LAWA / SWA refresh, dev/soup_avg.py);
#   cfg1 phf  = (phe + phf2) / 2 (two-window soup with phe as the anchor). Both with SL v4w and nav_avg=1e-4.
# - Base = phe pair files (rl3_policy_phe a1b8579c + rl2_sl_phe v4w) + nav_avg=1e-4, i.e. the finale file's weights.
# - 3,072 seeds 10062000-10065071 (set k, hour index i: 10062000 + 384 k + 48 i .. + 47); pack_phc.sh's official-check
#   seeds 10065100-10065115 are checked here too (registry max was 10061071).
# - Pool: P=phf S0=10062000 W1=phf W2=phf2 BN=phe BF=bpc_s40_fin_phe.py bash dev/pool_phc.sh; rule unchanged (>= +0.04,
#   z >= 2.24, no hour below -0.15, else phe). Pack: OFF_S0=10065100 BN=phe W=<phf|phf2> P=phf bash dev/pack_phc.sh.
#   Only once (an existing soup or probe_phfa dir aborts).   DRY=1 bash dev/launch_phf.sh      bash dev/launch_phf.sh
set -euo pipefail
cd ~/bpc
O=runs/oracle; V4F=runs/sl40/rl2_sl_s40lmn_v4w.npz
V4=49feb737c5b430c8c937ac0a506ead0b; PHE=a1b8579c920c565f2b473fd9aa858a68
declare -A END=([s40l]=651662880 [s40m]=647857191 [s40n]=668954910)  # phe's soup windows ended here (§497)
S0=10062000; NS=3072; MINE=${MINE:-48}
md5() { md5sum < "$1" | cut -c1-32; }
[ -n "${DRY:-}" ] || { [ ! -e $O/rl3_policy_phf.npz ] && [ ! -e runs/.cmp_parts/probe_phfa ] || { echo "ABORT: Phase F already built or launched"; exit 1; }; }
n=$(cut -d, -f1 data/seed_registry/used_seeds.csv | grep -E '^[0-9]+$' | awk -v a=$S0 -v b=$((S0 + NS + 116)) '$1 >= a && $1 < b' | wc -l)
[ "$n" -eq 0 ] || { echo "ABORT: seed registry overlaps [$S0, $((S0 + NS + 116)))"; exit 1; }
[ "$(md5 $O/rl3_policy_phe.npz)" = $PHE ] && [ "$(md5 $O/rl2_sl_phe.npz)" = $V4 ] && [ "$(md5 $V4F)" = $V4 ] \
  || { echo "ABORT: phe / v4w md5"; exit 1; }
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
  echo "$r: ${NE[$r]} exports after phe's window end ${END[$r]}, all v4w; newest ${NEW[$r]}"
  [ ${NE[$r]} -ge $MINE ] || { echo "ABORT: $r has only ${NE[$r]} exports after phe's window (< $MINE)"; exit 1; }
done
M=128; for r in s40l s40m s40n; do if [ ${NE[$r]} -lt $M ]; then M=${NE[$r]}; fi; done
echo "soup phf2: s40l s40m s40n, the newest $M exports of each; phf = (phe + phf2) / 2"
[ -z "${DRY:-}" ] || { echo "DRY: stop before writing"; exit 0; }
P=.venv/bin/python
nice -n 19 $P dev/soup_avg.py $O/rl3_policy_phf2.npz $M s40l:${NEW[s40l]} s40m:${NEW[s40m]} s40n:${NEW[s40n]}
nice -n 19 $P - <<'PY'
import numpy as np
L = lambda n: dict(np.load(f"runs/oracle/rl3_policy_{n}.npz", allow_pickle=True))
a, b = L("phe"), L("phf2")
assert sorted(a) == sorted(b), set(a) ^ set(b)
# same rule as dev/soup_avg.py: float arrays with ndim >= 1 averaged, 0-d / non-float keys from the first (phe)
np.savez("runs/oracle/rl3_policy_phf.npz", **{k: (np.mean([a[k], b[k]], axis=0).astype(a[k].dtype)
         if a[k].dtype.kind == "f" and a[k].ndim >= 1 else a[k]) for k in a})
print("runs/oracle/rl3_policy_phf.npz <- (rl3_policy_phe + rl3_policy_phf2) / 2")
PY
cp $V4F $O/rl2_sl_phf.npz; cp $V4F $O/rl2_sl_phf2.npz
[ "$(md5 $O/rl2_sl_phf.npz)" = $V4 ] && [ "$(md5 $O/rl2_sl_phf2.npz)" = $V4 ] || { echo "ABORT: pair file md5"; exit 1; }
md5sum $O/rl3_policy_phf.npz $O/rl2_sl_phf.npz $O/rl3_policy_phf2.npz $O/rl2_sl_phf2.npz \
  BalloonPoppingGymEnv/agents/{gnc_agent,rl2_agent,dp_plan,rl3_agent,rl2_sl}.py | tee $O/phf.md5
md5sum dev/sweep_gnc.py dev/probe_cmp.sh dev/line_cmp.sbatch
$P - <<'PY'
import numpy as np
L = lambda n: dict(np.load(f"runs/oracle/rl3_policy_{n}.npz", allow_pickle=True))
phe = L("phe")
f = [k for k, v in phe.items() if v.dtype.kind == "f" and v.ndim >= 1]
for n in ("phf", "phf2"):
    s = L(n)
    assert sorted(s) == sorted(phe), set(s) ^ set(phe)
    assert all(s[k].shape == phe[k].shape and np.isfinite(s[k]).all() for k in f)
    print(n, "non-averaged keys that differ from phe:", {k: (s[k].tolist(), phe[k].tolist()) for k in phe if k not in f and not np.array_equal(s[k], phe[k])})
    nrm = np.sqrt(sum(((s[k].astype(float) - phe[k].astype(float)) ** 2).sum() for k in f) / sum((phe[k].astype(float) ** 2).sum() for k in f))
    print(f"|{n} - phe| / |phe| = {nrm:.4f}")
PY
C1="policy_path=$O/rl3_policy_phf.npz sl_path=$O/rl2_sl_phf.npz"
C2="policy_path=$O/rl3_policy_phf2.npz sl_path=$O/rl2_sl_phf2.npz"
for k in 0 1 2 3 4 5 6 7; do
  AGENT_CLS=rl3 TAG=phf$(echo abcdefgh | cut -c$((k + 1))) HOURS="0 3 6 9 12 15 18 21" N=48 S0=$((S0 + 384 * k)) CORES=8 MEM=8G \
    TIME=1:00:00 EXCLUDE=cp11,cp14,cp18,cp20 BASE=$O/rl3_policy_phe.npz BASE_EXTRA=nav_avg=1e-4 CFGS="$C1;$C2" \
    setsid nohup bash dev/probe_cmp.sh >> logs/probe_sched.log 2>&1 < /dev/null &
done
sleep 30
squeue -u $USER -h -n bpc-cmp -o "%T" | sort | uniq -c || true

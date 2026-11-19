#!/usr/bin/env bash
# DECISION_LOG §460 Phase C packaging, only for the config dev/pool_phc.sh's VERDICT names (phc = cfg1 v4 soup + v4, curv4 =
# cfg2 cur + v4): single file BalloonPoppingGymEnv/agents/bpc_s40_fin_$W.py = runs/oracle/rl3_policy_$W.npz + its pair SL
# rl2_sl_$W.npz (v4w) + nav_avg=1e-4 baked in. The pair files and agent sources must still match runs/oracle/phc.md5
# (written by launch_phc.sh), and the embedded weights and SL are decoded back and checked against the npz md5s. Then the
# official evaluate.py vs sweep_gnc on one node, 8 hours x 2 fresh seeds 10052100-10052115 (offeval_8h.sbatch): 16/16
# identical by 11/7 08:00, else the finale keeps cur. RERUN=<failed offeval job id> resubmits on the same 16 seeds.
#   W=phc bash dev/pack_phc.sh          §480 Phase D: W=curv5 P=phd bash dev/pack_phc.sh (same offeval seeds, unused by Phase C)
#   §502 Phase F: OFF_S0=10065100 BN=phe W=<phf|phf2> P=phf bash dev/pack_phc.sh (10052100-10052115 were used by phe's check)
set -euo pipefail
cd ~/bpc
W=${W:?W=phc or W=curv4}; P=${P:-phc}; O=runs/oracle; F=BalloonPoppingGymEnv/agents/bpc_s40_fin_$W.py; S0=${OFF_S0:-10052100}
grep -q "^VERDICT: cfg[12] replaces ${BN:-cur} -> W=$W " logs/probe_${P}_all.log || { echo "ABORT: logs/probe_${P}_all.log has no VERDICT for W=$W"; exit 1; }
md5sum --quiet -c $O/$P.md5 || { echo "ABORT: pair files or agent sources changed since launch_$P.sh"; exit 1; }
if [ -n "${RERUN:-}" ]; then
  [ "$(head -1 logs/offeval_$RERUN.plan)" = "0 $S0 $((S0 + 1))" ] && grep -qx "agent_module_path: $F" logs/offeval_$RERUN.cfg.yaml \
    || { echo "ABORT: offeval $RERUN was not this file on these seeds"; exit 1; }
else
  n=$(cut -d, -f1 data/seed_registry/used_seeds.csv | grep -E '^[0-9]+$' | awk -v a=$S0 -v b=$((S0 + 16)) '$1 >= a && $1 < b' | wc -l)
  [ "$n" -eq 0 ] || { echo "ABORT: seed registry overlaps [$S0, $((S0 + 16))) (after a failed offeval of this file: RERUN=<job id>)"; exit 1; }
fi
nice -n 19 .venv/bin/python dev/make_single_file.py $O/rl3_policy_$W.npz $F $O/rl2_sl_$W.npz --kw "nav_avg=0.0001"
.venv/bin/python - $F $O/rl3_policy_$W.npz $O/rl2_sl_$W.npz <<'PY'
import base64, hashlib, re, sys
src = open(sys.argv[1]).read()
for var, f in (("_WEIGHTS_B64", sys.argv[2]), ("_SL_B64", sys.argv[3])):
    emb = hashlib.md5(base64.b64decode(re.search(rf'^{var} = "([^"]*)"', src, re.M)[1])).hexdigest()
    ref = hashlib.md5(open(f, "rb").read()).hexdigest()
    assert emb == ref, (var, emb, ref)
    print(f"{var}: embedded = {f} ({emb})")
assert re.search(r"^_BAKED_KW = \{'nav_avg': 0\.0001\}$", src, re.M), "nav_avg not baked"
print("_BAKED_KW nav_avg 0.0001")
PY
FILE=$F NPZ=$O/rl3_policy_$W.npz EXTRA=nav_avg=1e-4 S0=$S0 \
  sbatch -q normal -c 8 --mem=16G --exclude=cp11,cp14,cp18,cp20 --export=ALL dev/qual_check/offeval_8h.sbatch
echo "16/16 needed (§460; Phase F by 11/8 12:00, §502): logs/offeval_<job id>.log"

#!/usr/bin/env bash
# DECISION_LOG §508 packaging of the probability ensemble, only when dev/pool_phc.sh's VERDICT names it (P=pc, W=pqp):
# single file BalloonPoppingGymEnv/agents/bpc_s40_fin_pqp.py = rl3_policy_phel (main) + members phem, phen (--ens, the
# probe's ens= order) + SL v4w + nav_avg=1e-4 and ens_agg=prob baked in. The pair files and agent sources must still match
# runs/oracle/pc.md5 (launch_pc.sh); the embedded weights, members and SL are decoded and checked against the npz md5s.
# Then the official evaluate.py vs sweep_gnc on one node, 8 hours x 2 fresh seeds 10070100-10070115 (offeval_8h.sbatch):
# 16/16 identical before 11/8 11:30, else the finale keeps phe. RERUN=<failed offeval job id> resubmits on the same seeds.
#   bash dev/pack_pc.sh
#   DRY=1 OFF_S0=<16 used seeds> bash dev/pack_pc.sh   (rehearsal: no VERDICT / registry check, bpc_s40_dry_pqp.py, not committed)
set -euo pipefail
cd ~/bpc
O=runs/oracle; P=pc; W=pqp; S0=${OFF_S0:-10070100}; ENS=$O/rl3_policy_phem.npz,$O/rl3_policy_phen.npz
if [ -z "${DRY:-}" ]; then
  F=BalloonPoppingGymEnv/agents/bpc_s40_fin_$W.py
  grep -q "^VERDICT: cfg1 replaces phe -> W=$W " logs/probe_${P}_all.log || { echo "ABORT: logs/probe_${P}_all.log has no VERDICT for W=$W"; exit 1; }
else
  F=BalloonPoppingGymEnv/agents/bpc_s40_dry_$W.py; [ -n "${OFF_S0:-}" ] || { echo "ABORT: DRY needs OFF_S0 (16 used seeds)"; exit 1; }
fi
md5sum --quiet -c $O/$P.md5 || { echo "ABORT: pair files or agent sources changed since launch_pc.sh"; exit 1; }
if [ -n "${RERUN:-}" ]; then
  [ "$(head -1 logs/offeval_$RERUN.plan)" = "0 $S0 $((S0 + 1))" ] && grep -qx "agent_module_path: $F" logs/offeval_$RERUN.cfg.yaml \
    || { echo "ABORT: offeval $RERUN was not this file on these seeds"; exit 1; }
elif [ -z "${DRY:-}" ]; then
  n=$(cut -d, -f1 data/seed_registry/used_seeds.csv | grep -E '^[0-9]+$' | awk -v a=$S0 -v b=$((S0 + 16)) '$1 >= a && $1 < b' | wc -l)
  [ "$n" -eq 0 ] || { echo "ABORT: seed registry overlaps [$S0, $((S0 + 16))) (after a failed offeval of this file: RERUN=<job id>)"; exit 1; }
fi
nice -n 19 .venv/bin/python dev/make_single_file.py $O/rl3_policy_phel.npz $F $O/rl2_sl_phel.npz --kw "nav_avg=0.0001 ens_agg=prob" --ens $ENS
.venv/bin/python - $F $O/rl3_policy_phel.npz $O/rl2_sl_phel.npz ${ENS//,/ } <<'PY'
import ast, base64, hashlib, re, sys
src = open(sys.argv[1]).read()
md5, ref = lambda b: hashlib.md5(b).hexdigest(), lambda f: hashlib.md5(open(f, "rb").read()).hexdigest()
for var, f in (("_WEIGHTS_B64", sys.argv[2]), ("_SL_B64", sys.argv[3])):
    emb = md5(base64.b64decode(re.search(rf'^{var} = "([^"]*)"', src, re.M)[1]))
    assert emb == ref(f), (var, emb, ref(f))
    print(f"{var}: embedded = {f} ({emb})")
ens = ast.literal_eval(re.search(r"^_ENS_B64 = (\[.*?\])  #", src, re.M)[1])
assert [md5(base64.b64decode(b)) for b in ens] == [ref(f) for f in sys.argv[4:]], "members differ"
print("_ENS_B64: " + ", ".join(f"{f} ({ref(f)})" for f in sys.argv[4:]))
assert re.search(r"^_BAKED_KW = \{'nav_avg': 0\.0001, 'ens_agg': 'prob'\}$", src, re.M), "baked kwargs"
print("_BAKED_KW nav_avg 0.0001, ens_agg prob")
PY
FILE=$F NPZ=$O/rl3_policy_phel.npz EXTRA="nav_avg=1e-4 ens=$ENS ens_agg=prob" S0=$S0 \
  sbatch -q normal -c 8 --mem=16G --exclude=cp11,cp14,cp18,cp20 --export=ALL dev/qual_check/offeval_8h.sbatch
echo "16/16 needed before 11/8 11:30 (§508): logs/offeval_<job id>.log"

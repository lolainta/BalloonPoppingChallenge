#!/usr/bin/env bash
#SBATCH -p devq -J bpc-offeval --time=1:00:00
#SBATCH -o logs/offeval_%j.log
# Official evaluate.py vs sweep_gnc (the module) on the same node, 2 seeds per finale hour (DECISION_LOG §452): a single-file
# agent from make_single_file.py must fly every episode exactly like the npz it was built from. Same node, because CPUs with
# other numerics (cp20 vs cp11/13/15/16/17, §452) send the chaotic flights apart.
#   FILE=BalloonPoppingGymEnv/agents/bpc_s40_fb_cur.py NPZ=runs/oracle/rl3_policy_cur.npz EXTRA=nav_avg=1e-4 S0=<16 fresh seeds> \
#     sbatch -q normal -c 16 --mem=40G --exclude=cp14,cp18 --export=ALL dev/qual_check/offeval_8h.sbatch
cd ~/bpc
PY=/tmp/llt_py/.venv/bin/python; . dev/node_boot.sh
R=logs/offeval_$SLURM_JOB_ID
export PY R NPZ EXTRA SC=40 CFG=$R.cfg.yaml
sed "s|^agent_module_path: .*|agent_module_path: $FILE|; s|^agent_name: .*|agent_name: \"$(basename $FILE .py) (8-hour check)\"|" \
  dev/qual_check/final_cfg.yaml > $CFG  # secret-free (no team fields, leaderboard_submission false)
grep -qx "agent_module_path: $FILE" $CFG || { echo "ABORT: cfg $CFG lacks agent_module_path: $FILE"; exit 1; }
hostname; md5sum $FILE $NPZ dev/qual_check/eval_seed.py BalloonPoppingGymEnv/evaluation/evaluate.py
i=0; for h in 0 3 6 9 12 15 18 21; do echo "$h $((S0 + 2 * i)) $((S0 + 2 * i + 1))"; i=$((i + 1)); done > $R.plan
off() { HOUR=$1 $PY dev/qual_check/eval_seed.py $2 > $R.$2.out 2>&1; }
mod() { AGENT=rl3 $PY dev/sweep_gnc.py --scenario 40 --seeds $2-$3 --procs 2 --csv $R.h$1.csv --base policy_path=$NPZ \
          sl_path=${NPZ/rl3_policy/rl2_sl} P.date=2022,7,10,$1 ${EXTRA:-} --cfg "" > /dev/null 2>&1; }
export -f off mod
awk '{print $1, $2; print $1, $3}' $R.plan | xargs -P 8 -L 1 bash -c 'off "$@"' _
xargs -P 8 -L 1 bash -c 'mod "$@"' _ < $R.plan
$PY - $R <<'PY'
import csv, glob, re, sys
R = sys.argv[1]
off, mod = {}, {}
for f in glob.glob(f"{R}.*.out"):
    for line in open(f):
        m = re.match(r"RESULT seed (\d+) hour (\d+) pops (\d+) last_pop (\S+)", line)
        if m:
            off[int(m[1])] = (int(m[3]), None if m[4] == "None" else float(m[4]))
for f in glob.glob(f"{R}.h*.csv"):
    for r in csv.DictReader(open(f)):
        mod[int(r["seed"])] = (int(float(r["pops"])), float(r["last_pop"]) if r["last_pop"] else None, int(r["truncated"]))
ok = n = 0
for h, *seeds in (l.split() for l in open(f"{R}.plan")):
    for s in map(int, seeds):
        o, m = off.get(s), mod.get(s)
        same = (o is not None and m is not None and not m[2] and o[0] == m[0]
                and (o[1] is None) == (m[1] is None) and (o[1] is None or abs(o[1] - m[1]) < 0.006))
        n += 1; ok += same
        print(f"hour {int(h):2d} seed {s}: evaluate.py {o} | sweep_gnc {m} -> {'SAME' if same else 'DIFF'}")
print(f"{ok}/{n} identical")
PY

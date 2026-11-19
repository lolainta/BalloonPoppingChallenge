#!/bin/bash
# Scenario-4 search teacher job (DECISION_LOG §191): RUN (run dir; its newest checkpoint and SL), OUT (default $RUN/demos_in),
# NSEEDS (default 2 per core), TEACH_BEAM / TEACH_TOPK / TEACH_LAUNCH (dev/s4_teacher.py). Seeds are drawn like the
# workers' training episodes ([100000, 10000000)) and recorded in the seed registry before anything flies.
#SBATCH -p devq -J bpc-s4teach --time=6:00:00 --nice=1000
#SBATCH -o logs/s4teach_%j.log
cd ~/bpc
PY=${PY:-/tmp/llt_py/.venv/bin/python}
N=${NSEEDS:-$((SLURM_CPUS_PER_TASK * 2))}
SEEDS=$($PY -c "import random; r = random.Random(); print(' '.join(str(r.randrange(100000, 10000000)) for _ in range($N)))")
$PY -c "import sys; sys.path.insert(0, '.'); from dev.seed_registry import record; record([int(s) for s in '$SEEDS'.split()], '$(basename ${TEACHER:-dev/s4_teacher.py} .py)', note='$RUN job $SLURM_JOB_ID', scenario=4)"
echo "$(date +%T) $RUN seeds: $SEEDS"
# one process per seed (2026-10-16, §204): a Pool hangs for good once the cgroup OOM-kills a child (job 783728 held 69
# cores 2 h with 11 dead children); each seed also starts from the newest checkpoint at its start
# TEACHER: dev/s4_rollout_teacher.py = the teacher that never sees the episode's noise (§210)
echo $SEEDS | tr ' ' '\n' | xargs -P $SLURM_CPUS_PER_TASK -I{} $PY ${TEACHER:-dev/s4_teacher.py} $RUN ${OUT:-$RUN/demos_in} {}

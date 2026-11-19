#!/bin/bash
# One task of a qual_select.py sweep: its seed chunk, every candidate flown on the same reset (dev/sweep_gnc.py).
# Submitted by dev/qual_select.py submit (env QTAG QLO QHI QCHUNK). Local test of task 0 from the repo root:
#   QTAG=x QLO=10000 QHI=10001 QCHUNK=2 SLURM_ARRAY_TASK_ID=0 PY=.venv/bin/python bash dev/qual_eval.sbatch
#SBATCH -p devq -J bpc-qual -c 16 --mem-per-cpu=2G --time=24:00:00 --nice=1000
cd "${SLURM_SUBMIT_DIR:-.}"
D=logs/qual_$QTAG; i=$SLURM_ARRAY_TASK_ID
lo=$((QLO + i * QCHUNK)); hi=$((lo + QCHUNK - 1)); [ $hi -gt $QHI ] && hi=$QHI
mapfile -t CFG < $D/cfgs.txt  # one --cfg entry per candidate (policy_path / sl_path / kwargs), cands.txt order
echo "task $i seeds $lo-$hi, ${#CFG[@]} candidates, $(hostname) $(date)"
${PY:-$HOME/.local/bin/uv run python} dev/sweep_gnc.py --scenario 4 --seeds $lo-$hi --procs ${SLURM_CPUS_PER_TASK:-4} \
    --cfg "${CFG[@]}" --csv $D/part_$i.csv
echo "done $(date)"

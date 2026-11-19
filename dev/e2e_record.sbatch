#!/bin/bash
# Hybrid E2E imitation data (DECISION_LOG §257 / §260): the teacher flies (or, with E2E_NET, the net flies and the teacher labels
# in the shadow = DAgger) the seeds listed in SEEDFILE -> OUT/<seed>.npz; all the job's cores.
#   sbatch -q normal -w cp14 -c 48 --mem=96G --export=ALL,OUT=runs/e2e_bc/dagger1,SEEDFILE=runs/e2e_bc/dagger1.seeds,E2E_NET=runs/e2e_bc/bc_ptr.pt dev/e2e_record.sbatch
#SBATCH -p devq -J bpc-e2erec --time=3:00:00
#SBATCH -o logs/e2e_record_%j.log
cd ~/bpc
E2E_KW="${E2E_KW:-policy_path=runs/s4ne6/rl3_policy_381960016.npz sl_path=runs/s4ne6/rl2_sl_381960016.npz att_fix=0.07}" PROCS=$SLURM_CPUS_PER_TASK \
  /tmp/llt_py/.venv/bin/python dev/e2e.py record $OUT $(cat $SEEDFILE | tr '\n' ' ')

#!/bin/bash
# Plan-B gate (dev/gnc2_gate/gate.py): array task i of NT flies seeds 34000-34191 [i::NT] (free + all variants).
# ssh user@login 'cd ~/bpc && sbatch -p devq --qos=devq_low -c 8 --mem=14G --time=3:00:00 -J bpc-gate --array=0-11 --export=ALL,NT=12 -o logs/gate_%A_%a.out dev/gnc2_gate/gate.sbatch'
# VARIANTS=Pb:Fd:... (":"-separated; "," would break --export)
cd ~/bpc
md5sum BalloonPoppingGymEnv/agents/{gnc_agent,rl2_agent,rl3_agent,rl2_sl}.py runs/s4bw4/rl3_policy_25670400.npz runs/s4bw4/rl2_sl_25670400.npz
export OMP_NUM_THREADS=1
.venv/bin/python dev/gnc2_gate/gate.py run --seeds ${SEEDS:-34000-34191} --task $SLURM_ARRAY_TASK_ID --ntasks ${NT:-12} --procs ${SLURM_CPUS_PER_TASK:-8} --tag ${TAG:-gate} ${VARIANTS:+--variants ${VARIANTS//:/,}}

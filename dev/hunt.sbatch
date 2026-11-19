#!/bin/bash
# Cluster seed hunt (replaces dev/local_seed_hunt.sh): array task i = seeds [96i, 96i+96) of SF (one comma-separated line),
# POLICY P (+ its paired SL) with extra kwargs EXTRA -> logs/eval_<TAG>_<i>.log, the eval-job format dev/candidates.py
# collects (dev/local_services.sh mirrors cluster logs/ to the local host, which verifies and records videos).
#   sbatch --qos=devq_low --array=0-28 --export=ALL,P=runs/late3/rl3_policy_X.npz,SF=runs/hunt_seeds.txt,TAG=t,EXTRA="t_min=25" dev/hunt.sbatch
#SBATCH -p devq -J bpc-hunt -c 16 --mem=28G --time=2:00:00 --nice=1000
#SBATCH -o logs/hunt_%A_%a.out
cd ~/bpc
i=$SLURM_ARRAY_TASK_ID; f=logs/eval_${TAG}_$i.log
[ -s $f ] && exit 0
IFS=, read -ra S < $SF; chunk=$(IFS=,; echo "${S[*]:$((i * 96)):96}")
[ -n "$chunk" ] || exit 0
kw="policy_path=$P sl_path=${P/rl3_policy/rl2_sl} $EXTRA"
out=$(AGENT=rl3 .venv/bin/python dev/run_seeds.py --seeds $chunk --kw $kw 2>/dev/null)
{ echo "variant: $kw"; echo "agent rl3"; echo "$out"; } > $f.tmp && mv $f.tmp $f

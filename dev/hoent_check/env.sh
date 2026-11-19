# source me: the s4res2 training line's settings (deployed env), this tree on PYTHONPATH (the venv's editable install points
# at the main checkout), a tag for killing our own processes. RL3_HOENT / RL3_NEWENT / SIL_MIN are set by the caller.
W=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
S=./runs/s4res2
T=${HO_TMP:-$TMPDIR/ho}
export PYTHONPATH=$W HOENT_TAG=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= RL3_SCEN=4 RL3_ATTC=1 RL3_T_MIN=1 RL3_RES=0.25 RL3_RES2=1 \
  RL3_TCOND=1 RL3_FREE_LAUNCH=1 RL3_KEEP_LAUNCH=1 RL3_WAIT_NODISCOUNT=1 RL3_FASTWIND=1 \
  SLMODEL=$S/rl2_sl_122313600.npz SEED_REGISTRY=$T/seed_registry.csv SEED_LIST="10000 10001 10002 10003" \
  GNC_KW="k_att:3+max_rate:2.5+coast:1+min_alt:3+lag_comp:1+bwind:4+margin:0+eob:2+eob_tau:3+eob_T:4"
PY=./.venv/bin/python
GPY=./.venv-gpu/bin/python
POL=$S/rl3_policy_122313600.npz

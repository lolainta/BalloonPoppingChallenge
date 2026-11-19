# source me: the s4arrv training line's settings (+ RL3_TCOND2 from the caller), a tree on PYTHONPATH (default: this one; the
# venv's editable install points at the main checkout), a tag for killing our own processes, a temp seed registry
W=${TC2_TREE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
T=${TC2_TMP:-$TMPDIR/tc2}
export PYTHONPATH=$W TC2_TAG=1 OMP_NUM_THREADS=1 RL3_SCEN=4 RL3_ATTC=1 RL3_T_MIN=1 RL3_RES=0.25 RL3_RES2=1 RL3_RES3=1 RL3_TCOND=1 \
  RL3_TCOND2=${RL3_TCOND2-1} SEED_REGISTRY=$T/seed_registry.csv \
  GNC_KW="k_att:3+max_rate:2.5+coast:1+min_alt:3+lag_comp:1+bwind:4+margin:0+eob:2+eob_tau:3+eob_T:4"
PY=./.venv/bin/python
R=./runs
R8=$R/s4arrv/rl3_policy_132854400.npz; SL8=$R/s4arrv/rl2_sl_132854400.npz

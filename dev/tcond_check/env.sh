# source me: the s4res2 training line's settings + RL3_TCOND, this tree on PYTHONPATH (the venv's editable install points
# at the main checkout), a tag for killing our own processes
W=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export PYTHONPATH=$W TCOND_TAG=1 OMP_NUM_THREADS=1 RL3_SCEN=4 RL3_ATTC=1 RL3_T_MIN=1 RL3_RES=0.25 RL3_RES2=1 RL3_TCOND=1 \
  GNC_KW="k_att:3+max_rate:2.5+coast:1+min_alt:3+lag_comp:1+bwind:4+margin:0+eob:2+eob_tau:3+eob_T:4"
PY=./.venv/bin/python
T=${TC_TMP:-$TMPDIR/ck}

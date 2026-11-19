#!/usr/bin/env bash
# A/B/C/F: per-step action hashes (scenario 4).   [ARRV_TREE=<tree>] dev/arrv_check/hashes.sh A|B|C|F [k=v ...]
#   A: no-head / 3-output / 5-output / 5-output+tc checkpoints (pre-change reference: ARRV_TREE = a `git archive` of the base)
#   B: the 5-output(+tc) checkpoints expanded to 8 outputs, zero new rows (make_arrv_npz.py)
#   C: the tc one with non-zero u[5:8] rows;  F: that one with res_vbound=0
source "$(dirname "$0")/env.sh"; cd $W
S5=$R/s4res2/rl3_policy_122313600.npz@$R/s4res2/rl2_sl_122313600.npz
case $1 in
  A) J="$R/s4ent/rl3_policy_88761600.npz@$R/s4ent/rl2_sl_88761600.npz@10000 $R/s4res/rl3_policy_99561600.npz@$R/s4res/rl2_sl_99561600.npz@10001 $S5@10002 $TC@10002,10003";;
  B) J="$T/r8_s4res2.npz@$R/s4res2/rl2_sl_122313600.npz@10002 $T/r8_tc.npz@$T/rl2_sl_128851200.npz@10002,10003";;
  C|F) J="$T/r8_tc_dv.npz@$T/rl2_sl_128851200.npz@10002,10003";;
esac
[ "$1" = F ] && set -- F res_vbound=0 "${@:2}"
shift
nice -n 10 $PY $W/dev/arrv_check/act_hash.py $J "$@" 2>&1 | grep -v Warn

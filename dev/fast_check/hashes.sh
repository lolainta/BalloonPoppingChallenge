#!/usr/bin/env bash
# A/D: per-step action hashes (scenario 4).   [FAST_TREE=<tree>] dev/fast_check/hashes.sh A|Z|N|O [k=v ...]
#   A: s4ent (no head) 10000, s4arrv (8-output + tc) 10001, s4next (8-output + tc + tc2) 10002,10003 (pre-change reference:
#      FAST_TREE = a `git archive` of the base);  Z: s4next + a zero fast head (k = 10, make_fast_npz.py);  N: + a non-zero one;
#   O: N with fast_off=1
source "$(dirname "$0")/env.sh"; cd $W
J=
for c in $(echo $1 | grep -o .); do case $c in
  A) J+=" $R/s4ent/rl3_policy_88761600.npz@$R/s4ent/rl2_sl_88761600.npz@10000 $R/s4arrv/rl3_policy_142934400.npz@$R/s4arrv/rl2_sl_142934400.npz@10001 $RN@$SLN@10002,10003";;
  a) J+=" $R/s4ent/rl3_policy_88761600.npz@$R/s4ent/rl2_sl_88761600.npz@10000 $R/s4arrv/rl3_policy_142934400.npz@$R/s4arrv/rl2_sl_142934400.npz@10001 $RN@$SLN@10002";;  # A, 1 seed per kind
  Z) J+=" $T/fast_0.npz@$SLN@10002,10003";;
  N) J+=" $T/fast_1.npz@$SLN@10002,10003";;
  O) J+=" $T/fast_1.npz@$SLN@10002,10003 fast_off=1";;
esac; done
shift
nice -n 10 $PY $W/dev/fast_check/act_hash.py $J "$@" 2>&1 | grep -v Warn

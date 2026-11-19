#!/usr/bin/env bash
# A/B/C/E: per-step action hashes (scenario 4).   [TC2_TREE=<tree>] dev/tc2_check/hashes.sh A|B|C|E [k=v ...]
#   A: no-head / 5-output+tc / 8-output+tc checkpoints (pre-change reference: TC2_TREE = a `git archive` of the base)
#   B: the 8-output+tc one + a zero tc2 head (make_tc2_npz.py);  C: + a non-zero tc2 head;  E: that one with tc2_off=1
source "$(dirname "$0")/env.sh"; cd $W
J=  # several letters (e.g. AB): one pool (E's kwarg applies to all)
for c in $(echo $1 | grep -o .); do case $c in
  A) J+=" $R/s4ent/rl3_policy_88761600.npz@$R/s4ent/rl2_sl_88761600.npz@10000 $R/s4tc/rl3_policy_132681600.npz@$R/s4tc/rl2_sl_132681600.npz@10001 $R8@$SL8@10002,10003";;
  B) J+=" $T/tc2_0.npz@$SL8@10002,10003";;
  C) J+=" $T/tc2_1.npz@$SL8@10002,10003";;
  E) J+=" $T/tc2_1.npz@$SL8@10002,10003 tc2_off=1";;
esac; done
shift
nice -n 10 $PY $W/dev/tc2_check/act_hash.py $J "$@" 2>&1 | grep -v Warn

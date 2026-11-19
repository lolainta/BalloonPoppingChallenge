#!/usr/bin/env bash
# A/B/C(i)/E: per-step action hashes (scenario 4) of this tree.   dev/tcond_check/hashes.sh A|B|C|E [k=v ...]
#   A: the three checkpoint kinds (pre-change reference: act_hash.py run on a `git archive` of the base commit)
#   B: the 5-output checkpoint + a zero tc head;  C: + a non-zero tc head (make_tc_npz.py);  E: that one with tc_off=1
source "$(dirname "$0")/env.sh"; cd $W
R=./runs
S5=$T/rl3_policy_116668800.npz@$T/rl2_sl_116668800.npz
case $1 in
  A) J="$R/s4ent/rl3_policy_88761600.npz@$R/s4ent/rl2_sl_88761600.npz@10000 $R/s4res/rl3_policy_99561600.npz@$R/s4res/rl2_sl_99561600.npz@10001 $S5@10002,10003";;
  B) J="$T/tc0.npz@$T/rl2_sl_116668800.npz@10002,10003";;
  E) J="$T/tc1.npz@$T/rl2_sl_116668800.npz@10002,10003 tc_off=1";;
  C) J="$T/tc1.npz@$T/rl2_sl_116668800.npz@10002,10003";;
esac
shift
nice -n 10 $PY dev/tcond_check/act_hash.py $J "$@" 2>&1 | grep -v Warn

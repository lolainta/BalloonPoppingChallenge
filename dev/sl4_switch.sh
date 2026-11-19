#!/usr/bin/env bash
# DECISION_LOG §460: put SL file V4 live on the three s40 branches in place. Workers reload runs/<run>/rl2_sl.npz (SLMODEL) by
# mtime at their next episode (dev/rl2_env.py calls load_sl unguarded, so a bad file would crash them: V4 is loaded and run
# on a dummy batch first); the learner keeps running and copies the live file to rl2_sl_<steps>.npz at every export.
# All checks run before the first mv. runs/<run>/sl4_switch records the time, the md5 and the newest export at the switch;
# the exact switch point is the first export whose rl2_sl_<steps>.npz copy has the new md5.
#   V4=runs/sl40/rl2_sl_s40lmn_v4w.npz bash dev/sl4_switch.sh          revert: V4=runs/sl40/rl2_sl_s40a_v2.npz bash dev/sl4_switch.sh
set -euo pipefail
cd ~/bpc
V4=${V4:?set V4=<sl npz>}
.venv/bin/python -c "
import sys, numpy as np; sys.path.insert(0, '.')
from BalloonPoppingGymEnv.agents.rl2_sl import load_sl
t, p = load_sl('$V4')(np.zeros((5, 15), np.float32), np.zeros(6, np.float32))
assert t.shape == p.shape == (5,) and np.isfinite(t).all() and np.isfinite(p).all(), (t, p)
print('$V4 loads, dummy batch t', t.round(3), 'p', p.round(3))"
for r in s40l s40m s40n; do
  grep -q "SLMODEL=runs/$r/rl2_sl.npz" runs/$r/learner_submit.sh || { echo "ABORT: $r learner SLMODEL"; exit 1; }
  if grep "^runs/$r " runs/workers01.spec | grep -v "SLMODEL=runs/$r/rl2_sl.npz" | grep -q .; then echo "ABORT: $r worker SLMODEL"; exit 1; fi
done
for r in s40l s40m s40n; do
  cp "$V4" runs/$r/.rl2_sl.switch.tmp && mv -f runs/$r/.rl2_sl.switch.tmp runs/$r/rl2_sl.npz
  last=$(ls runs/$r/rl3_policy_[0-9]*.npz | sed 's/.*_//; s/\.npz//' | sort -n | tail -1)
  echo "$(date '+%F %T') $r: live SL $(md5sum < runs/$r/rl2_sl.npz | cut -c1-8) = $V4, newest export $last" | tee runs/$r/sl4_switch
done

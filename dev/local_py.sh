#!/usr/bin/env bash
# Copy the uv python + a venv to a node's local /tmp (run on login02): dev/local_py.sh <jobid on the node> [venv] [dest]
# (default .venv -> /tmp/llt_py, the workers' PY=/tmp/llt_py/.venv/bin/python; .venv-gpu -> /tmp/llt_gpu for a learner).
# Why (DECISION_LOG §164 / §167): processes that run python from NFS keep its .so files open there; on several nodes the NFSv4
# client then loops in TEST_STATEID (~380/s on cp17, 2026-10-05) and every op on ~/ takes 0.5-3 s. /tmp is RAM: redo after a reboot.
set -e
J=$1; V=${2:-.venv}; T=${3:-/tmp/llt_py}; PYD=cpython-3.14.5-linux-x86_64-gnu
tar -C ~/.local/share/uv/python -cf - $PYD | srun --overlap --jobid=$J -n1 -c1 bash -c "mkdir -p $T && tar -C $T -xf -"
tar -C ~/bpc -cf - $V | srun --overlap --jobid=$J -n1 -c1 bash -c "tar -C $T -xf -"
srun --overlap --jobid=$J -n1 -c1 bash -c "cd $T/$V/bin && for f in python python3 python3.14; do ln -sf $T/$PYD/bin/python3.14 \$f; done
  sed -i 's|^home = .*|home = $T/$PYD/bin|' ../pyvenv.cfg
  cd ~/bpc && s=\$(date +%s) && $T/$V/bin/python -c 'import numpy, scipy, rocketpy, dev.rl3_env' && echo \$(hostname -s) $T/$V ok, import \$((\$(date +%s) - s)) s"

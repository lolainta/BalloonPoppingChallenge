#!/bin/bash
# Copy the uv python + a venv to this node's /tmp from inside a job (DECISION_LOG §255): dev/local_py.sh pipes the tar
# through srun from the login node, too slow for .venv-gpu (3.8 GB: cut by a timeout twice, venv left without numpy).
# tar only reads the NFS files (no python runs from NFS). V (.venv-gpu) -> T (/tmp/llt_gpu); the job's memory must
# cover the tmpfs pages it writes (8 GB was not enough).
#   sbatch -q normal -w cp12 -c 2 --mem=16G --export=ALL,V=.venv-gpu,T=/tmp/llt_gpu dev/node_venv.sbatch
#SBATCH -p devq -J bpc-venv --time=1:00:00
#SBATCH -o logs/node_venv_%j.log
set -e
V=${V:-.venv-gpu}; T=${T:-/tmp/llt_gpu}; PYD=cpython-3.14.5-linux-x86_64-gnu
mkdir -p $T
tar -C ~/.local/share/uv/python -cf - $PYD | tar -C $T -xf -
tar -C ~/bpc -cf - $V | tar -C $T -xf -
cd $T/$V/bin && for f in python python3 python3.14; do ln -sf $T/$PYD/bin/python3.14 $f; done
sed -i "s|^home = .*|home = $T/$PYD/bin|" ../pyvenv.cfg
cd ~/bpc && $T/$V/bin/python -c "import numpy, scipy, torch, rocketpy, dev.rl3_env; print('$(hostname -s) $T/$V ok: numpy', numpy.__version__, 'torch', torch.__version__)"
du -sh $T/$V; df -h /tmp | tail -1

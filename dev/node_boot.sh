# Sourced by the sbatch scripts (DECISION_LOG §285): jobs are no longer pinned to nodes, so the node may lack the node-local
# copies (python must not run from NFS, §167) -> unpack them from ~/bpc_store (one big NFS read each) under a node lock.
# Usage after `cd ~/bpc`: . dev/node_boot.sh   (looks at $PY and $SCRIPT)
boot() { [ -e "$3" ] && return 0; mkdir -p "$1"; ( flock 9; [ -e "$3" ] || tar -C "$1" -xf "$2" ) 9>/tmp/llt_boot.lock; }
case "${PY:-}" in /tmp/llt_py/*) boot /tmp/llt_py ~/bpc_store/llt_py.tar "$PY" ;; /tmp/llt_gpu/*) boot /tmp/llt_gpu ~/bpc_store/llt_gpu.tar "$PY" ;; esac
case "${SCRIPT:-}" in /tmp/llt_gpu/code/*) boot /tmp/llt_gpu/code ~/bpc_store/llt_code.tar /tmp/llt_gpu/code/dev/async_ppo.py ;; esac
# the learner's code copy: refresh the .py files from NFS on every start (an old node copy would miss later fixes, §287)
case "${SCRIPT:-}" in /tmp/llt_gpu/code/*) rsync -a --include='*/' --include='*.py' --exclude='*' ~/bpc/dev ~/bpc/BalloonPoppingGymEnv /tmp/llt_gpu/code/ ;; esac

"""Uniform weight average of RL3 policy checkpoints (float weights averaged; ints and gnc.* / t_min copied from the last).
  dev/avg_ckpt.py <out.npz> <ckpt.npz> [<ckpt.npz> ...]"""
import sys
import numpy as np

out, paths = sys.argv[1], sys.argv[2:]
zs = [np.load(p) for p in paths]
keys = zs[-1].files
assert all(z.files == keys for z in zs), "checkpoints differ in keys"
res = {}
for k in keys:
    a = zs[-1][k]
    avg = a.dtype.kind == "f" and a.ndim > 0 and not k.startswith("gnc.")
    res[k] = np.mean([z[k] for z in zs], axis=0).astype(a.dtype) if avg else a
np.savez(out, **res)
print(out, "<-", len(paths), "checkpoints")


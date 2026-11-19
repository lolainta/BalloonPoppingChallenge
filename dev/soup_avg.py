#!/usr/bin/env python3
# DECISION_LOG §372: generic checkpoint soup. Usage:  soup_avg.py OUT.npz K run:step [run:step ...]
# For each run, averages its last K rl3_policy exports ending at step (float arrays with ndim>=1; 0-d / non-float
# taken from the newest), cached as runs/<run>/avgK_<step>.npz; then averages those per-run averages into OUT
# (0-d / non-float from the first run listed). One run = plain weight averaging; several = cross-branch model soup.
import numpy as np, glob, os, sys
def load(f): return np.load(f, allow_pickle=True)
def mean_of(dicts, newest):
    return {k: (np.mean([d[k] for d in dicts], axis=0).astype(newest[k].dtype) if newest[k].dtype.kind == 'f' and newest[k].ndim >= 1 else newest[k]) for k in newest.keys()}
def avg_k(run, st, k):
    out = f"runs/{run}/avg{k}_{st}.npz"
    if not os.path.exists(out):
        fs = sorted(glob.glob(f"runs/{run}/rl3_policy_[0-9]*.npz"), key=lambda f: int(f.split("_")[-1][:-4]))
        i = [int(f.split("_")[-1][:-4]) for f in fs].index(st); sel = fs[i - k + 1:i + 1]
        assert len(sel) == k, (run, st, len(sel))
        np.savez(out, **mean_of([load(f) for f in sel], load(sel[-1])))
    return out
out, k = sys.argv[1], int(sys.argv[2]); parts = [avg_k(r, int(s), k) for r, s in (a.split(":") for a in sys.argv[3:])]
ds = [load(p) for p in parts]; np.savez(out, **mean_of(ds, ds[0])); print(out, "<-", *parts)

# DECISION_LOG §371: cross-branch "model soup" files for a zero-training probe. Same recipe as the avg24 probes
# (float arrays with ndim>=1 averaged, 0-d / non-float taken from the newest), plus an averaged rl2_sl for the soups.
import numpy as np, glob
def load(f): return np.load(f, allow_pickle=True)
def avg(files, newest):
    ds = [load(f) for f in files]; new = load(newest); out = {}
    for k in new.keys():
        a = new[k]
        out[k] = np.mean([d[k] for d in ds], axis=0).astype(a.dtype) if a.dtype.kind == 'f' and a.ndim >= 1 else a
    return out
def last_k(run, st, k):
    fs = sorted(glob.glob(f"runs/{run}/rl3_policy_[0-9]*.npz"), key=lambda f: int(f.split("_")[-1][:-4]))
    i = [int(f.split("_")[-1][:-4]) for f in fs].index(st); sel = fs[i - k + 1:i + 1]
    assert len(sel) == k, (run, st, len(sel)); return sel
def dist(a, b):  # relative L2 distance over float arrays
    num = den = 0.0
    for k in a.keys():
        if a[k].dtype.kind == 'f' and a[k].ndim >= 1:
            x = a[k].astype(np.float64); y = b[k].astype(np.float64); num += ((x - y) ** 2).sum(); den += (x ** 2).sum()
    return (num / den) ** 0.5
L, M = 596760684, 596759708
fl, fm = f"runs/s40l/rl3_policy_{L}.npz", f"runs/s40m/rl3_policy_{M}.npz"
dl, dm = load(fl), load(fm); assert set(dl.keys()) == set(dm.keys())
for k in dl.keys(): assert dl[k].shape == dm[k].shape, k
print("policy rel dist L vs M", round(dist(dl, dm), 4))
np.savez(f"runs/s40l/soup_{L}.npz", **avg([fl, fm], fl))
a24l = avg(last_k("s40l", L, 24), fl); np.savez(f"runs/s40l/avg24_{L}.npz", **a24l)
a24m = avg(last_k("s40m", M, 24), fm); np.savez(f"runs/s40m/avg24_{M}.npz", **a24m)
print("policy rel dist avg24l vs avg24m", round(dist(a24l, a24m), 4), "| L vs avg24l", round(dist(dl, a24l), 4))
sa = {k: (((a24l[k].astype(np.float64) + a24m[k].astype(np.float64)) / 2).astype(a24l[k].dtype) if a24l[k].dtype.kind == 'f' and a24l[k].ndim >= 1 else a24l[k]) for k in a24l}
np.savez(f"runs/s40l/soupavg_{L}.npz", **sa)
sl, sm = f"runs/s40l/rl2_sl_{L}.npz", f"runs/s40m/rl2_sl_{M}.npz"
dsl, dsm = load(sl), load(sm); assert set(dsl.keys()) == set(dsm.keys())
print("sl rel dist L vs M", round(dist(dsl, dsm), 4))
np.savez(f"runs/s40l/soup_sl_{L}.npz", **avg([sl, sm], sl))
print("done")

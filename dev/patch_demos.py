# DECISION_LOG §195: add dt (from x[63] = t / 60) to teacher demos written without it; new name so the learner reads them
import glob, os, sys, numpy as np
n = 0
for f in sorted(glob.glob(os.path.join(sys.argv[1] if len(sys.argv) > 1 else "runs/s4teach", "demos_in", "[0-9]*.npz"))):
    if f.endswith("_dt.npz"):
        continue
    d = dict(np.load(f, allow_pickle=True))
    if "dt" in d:
        continue
    o = d["obs"]; d["dt"] = np.append(np.diff(o[:, 63].astype(np.float64)) * 60.0, 0.5).astype(np.float32)
    assert (d["dt"] > 0).all(), f
    out = f[:-4] + "_dt.npz"; tmp = os.path.join(os.path.dirname(f), "." + os.path.basename(out) + ".tmp.npz")
    np.savez(tmp, **d); os.replace(tmp, out); os.remove(f); n += 1
n and print("patched", n)

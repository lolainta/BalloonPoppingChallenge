"""Can "wait until 47.5 s instead of launching now" be predicted from the policy's launch-time features? (DECISION_LOG §227)
Data: dev/s4_rollout_teacher.py ROLL_K=0 demos (x_launch = the policy's launch decision features, launch_vals = mean pops
of the policy's launch and of 47.5 s over the same fresh noise draws). y = val(47.5) - val(policy).
10-fold cross-validated ridge regression; on each held-out seed: wait if the prediction > thr. The mean of y over the waited
seeds / all seeds is an honest estimate of the rule's gain (the model never saw that seed's y; y's noise averages out).
   .venv/bin/python dev/launch_probe.py <demo dir>"""
import glob, sys

import numpy as np

X, Y, T = [], [], []
for f in sorted(glob.glob(f"{sys.argv[1]}/[0-9]*.npz")):
    d = np.load(f)
    if "x_launch" not in d.files:
        continue
    o, v = list(d["launch_opts"]), d["launch_vals"]
    X.append(d["x_launch"]); Y.append(v[o.index(47.5)] - v[o.index(-1.0)]); T.append(float(d["t_launch"]))
X, Y, T = np.array(X, np.float64), np.array(Y), np.array(T)
n = len(Y); keep = X.std(0) > 1e-6; X = X[:, keep]
print(f"{n} seeds, {X.shape[1]} non-constant features; y mean {Y.mean():+.3f} sd {Y.std():.2f}; always wait {Y.mean():+.3f}, "
      f"oracle (noisy) {np.maximum(Y, 0).mean():+.3f}")
fold = np.random.default_rng(0).permutation(n) % 10


def cv(F, alpha):
    pred = np.zeros(n)
    for k in range(10):
        tr, te = fold != k, fold == k
        mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-9
        A, b = (F[tr] - mu) / sd, Y[tr] - Y[tr].mean()
        w = np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ b)
        pred[te] = (F[te] - mu) / sd @ w + Y[tr].mean()
    return pred


def report(name, pred):
    r = np.corrcoef(pred, Y)[0, 1]
    out = [f"{name:34s} r {r:+.3f}"]
    for thr in (0.0, 0.25, 0.5):
        g = Y * (pred > thr)
        out.append(f"thr {thr:.2f}: wait {np.mean(pred > thr):4.0%} gain {g.mean():+.3f} ± {g.std() / np.sqrt(n):.3f}")
    print("   ".join(out))


for alpha in (1e2, 1e3, 1e4, 1e5):
    report(f"ridge all features a={alpha:g}", cv(X, alpha))
report("ridge t_launch only", cv(np.c_[T, T ** 2], 1.0))

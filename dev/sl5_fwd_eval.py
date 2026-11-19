# DECISION_LOG §480 / §482 gate A'': the live SL v4w against v5w / v5b (dev/sl40_v5.sbatch) on decision logs written from
# 11/6 07:06 on (file-name time_ns >= 1793920000, after v5's data; the learners stopped at 10:03, so the set is fixed).
# Metrics at the logged stretch level as dev/sl_eval.py eval, plus a file-level bootstrap (rows of one file are correlated)
# of v5 - v4w.
# Pre-registered: a v5 becomes the eval-time candidate (Phase D, cur + v5 vs cur + v4 vs cur) only if its log loss is >= 1 %
# (relative) below v4w's and its AUC >= +0.005 above, both over the whole 95 % bootstrap interval; v5b over v5w only if
# it also beats v5w's log loss by >= 0.5 % (relative). Otherwise v4w stays and there is no Phase D.
#   .venv/bin/python dev/sl5_fwd_eval.py
import glob, os, sys
import numpy as np
sys.path.insert(0, "dev"); sys.path.insert(0, ".")
from sl_eval import outputs, auc  # noqa: E402

HOLD = "runs/s4data/t1data_s40[lmn]_v2/*_17939[2-9]*.npy"  # time_ns 1793920000.. = 11/6 07:06 on
parts, bad = [], 0
for f in sorted(glob.glob(HOLD), key=lambda f: int(os.path.basename(f).split("_")[1].split(".")[0])):
    try:
        parts.append(np.load(f))
    except ValueError:  # a log still being written when the links were made (train_t1sl.py skips those too)
        bad += 1
R = np.concatenate(parts); ends = np.cumsum([len(p) for p in parts]); starts = ends - [len(p) for p in parts]
y, t = R[:, 22], R[:, 23]; h = y > 0
print(f"{len(parts)} files ({bad} unreadable, skipped), {len(R)} rows, hit rate {y.mean():.4f}, mean t_hit {t[h].mean():.2f} s")
M = {"v2": "runs/sl40/rl2_sl_s40a_v2.npz", "v4w": "runs/sl40/rl2_sl_s40lmn_v4w.npz",
     "v5w": "runs/sl40/rl2_sl_s40lmn_v5w.npz", "v5b": "runs/sl40/rl2_sl_s40lmn_v5b.npz"}
out = {}
for k, path in M.items():
    p, th = outputs(path, R); pc = np.clip(p, 1e-6, 1 - 1e-6)
    out[k] = (p, -(y * np.log(pc) + (1 - y) * np.log(1 - pc)), np.where(h, np.abs(th - t), np.nan))
    print(f"  {k:<4} AUC {auc(p, y):.4f}  logloss {out[k][1].mean():.4f}  mean p {p.mean():.4f}  "
          f"t MAE on hits {np.nanmean(out[k][2]):.3f} s")
rng = np.random.default_rng(0)
q = lambda a: f"{np.mean(a):+.4f} [{np.quantile(a, 0.025):+.4f}, {np.quantile(a, 0.975):+.4f}]"  # noqa: E731
gate, ll = {}, {}
for k in ("v5w", "v5b"):
    d_auc, d_ll, d_t = [], [], []
    for _ in range(200):
        idx = np.concatenate([np.arange(starts[i], ends[i]) for i in rng.integers(0, len(parts), len(parts))])
        d_auc.append(auc(out[k][0][idx], y[idx]) - auc(out["v4w"][0][idx], y[idx]))
        d_ll.append(out[k][1][idx].mean() / out["v4w"][1][idx].mean() - 1)
        d_t.append(np.nanmean(out[k][2][idx]) - np.nanmean(out["v4w"][2][idx]))
    gate[k] = np.quantile(d_ll, 0.975) <= -0.01 and np.quantile(d_auc, 0.025) >= 0.005; ll[k] = out[k][1].mean()
    print(f"{k} - v4w: AUC {q(d_auc)}  logloss rel {q(d_ll)}  t MAE {q(d_t)} s -> gate A'' {'PASS' if gate[k] else 'FAIL'}")
pick = "v5b" if gate["v5b"] and (not gate["v5w"] or ll["v5b"] / ll["v5w"] - 1 <= -0.005) else "v5w" if gate["v5w"] else None
print("CANDIDATE:", f"{pick} -> Phase D (cur + {pick} vs cur + v4w vs cur)" if pick else "none, v4w stays, no Phase D")

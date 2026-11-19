"""Reach-model (SL) accuracy on held-out decision logs (DECISION_LOG §190).
  split: .venv/bin/python dev/sl_eval.py split <log dir> <train dir> <holdout dir> [holdout fraction]
         newest files (by the time_ns in pid_timens.npy) go to the holdout; symlinks, nothing is copied
  eval:  .venv/bin/python dev/sl_eval.py eval <holdout dir> <sl.npz> [<sl.npz> ...]
         per model, on the row's own stretch level: hit AUC, log loss, mean predicted p vs the hit rate, |t - t_hit| on hits
"""
import glob, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from BalloonPoppingGymEnv.agents.rl2_sl import IN_TOK


def files(d):
    return sorted(glob.glob(os.path.join(d, "*.npy")), key=lambda f: int(os.path.basename(f).split("_")[1].split(".")[0]))


def outputs(path, R):
    d = dict(np.load(path)); n = int(d["n_layers"]); ns = int(d["n_str"])
    z = (np.concatenate([R[:, :6], R[:, 6:21][:, IN_TOK]], 1) - d["mu"]) / d["sd"]
    for i in range(n - 1):
        z = np.maximum(z @ d[f"W{i}"] + d[f"b{i}"], 0)
    o = z @ d[f"W{n - 1}"] + d[f"b{n - 1}"]
    r = np.arange(len(R)); s = R[:, 21].astype(int)
    return 1 / (1 + np.exp(-o[r, s])), np.maximum(o[r, ns + s], 0.0)


def auc(p, y):
    o = np.argsort(p); rk = np.empty(len(p)); rk[o] = np.arange(1, len(p) + 1); n1 = y.sum(); n0 = len(y) - n1
    return (rk[y > 0].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "split":
        src, tr, ho = sys.argv[2:5]; frac = float(sys.argv[5]) if len(sys.argv) > 5 else 0.1
        fs = files(src); k = int(len(fs) * (1 - frac))
        for d, part in ((tr, fs[:k]), (ho, fs[k:])):
            os.makedirs(d, exist_ok=True)
            for f in part:
                dst = os.path.join(d, os.path.basename(f))
                if not os.path.exists(dst):
                    os.symlink(os.path.abspath(f), dst)
        print(f"train {k} files, holdout {len(fs) - k} files")
    elif cmd == "eval":
        R = np.concatenate([np.load(f) for f in files(sys.argv[2])])
        y, t = R[:, 22], R[:, 23]; h = y > 0
        print(f"holdout rows {len(R)}  hit rate {y.mean():.3f}  mean t_hit {t[h].mean():.2f} s")
        for path in sys.argv[3:]:
            p, th = outputs(path, R)
            pc = np.clip(p, 1e-6, 1 - 1e-6)
            ll = -np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))
            print(f"  {path:<58} AUC {auc(p, y):.3f}  logloss {ll:.3f}  mean p {p.mean():.3f}  t MAE on hits {np.abs(th[h] - t[h]).mean():.2f} s")

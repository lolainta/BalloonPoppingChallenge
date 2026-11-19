"""Common evaluation for balloon-trajectory predictors (dev/traj/make_dataset.py data).

A predictor is a class Predictor(given_parameters) with
    predict(states, status, t, idx, horizons) -> (len(idx), len(horizons), 3) predicted positions
where states (k+1, 100, 6) / status (k+1, 100) / t (k+1,) are ALL observations up to now (index k = now; an agent may
keep its history), idx = the balloons to predict, horizons in seconds. It may only use observations and the given
parameters (honor code: no environment internals, no seed). Learned predictors may train on TRAIN seeds.
    .venv/bin/python dev/traj/harness.py dev/traj/pred_x.py        (reports held-out TEST seeds)
"""
import glob, importlib.util, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data/traj/s4")
TRAIN = [f"{DATA}/seed{s}.npz" for s in range(20000, 20032)]
TEST = [f"{DATA}/seed{s}.npz" for s in range(20032, 20048)]
HORIZONS = np.array([0.5, 1.0, 2.0, 3.0, 4.0, 6.0])
QUERY_T = np.arange(6.0, 50.1, 2.0)  # query times (s)
GIVEN = load_scenario_parameters(4)[1]
ELEV = GIVEN["environment"]["elevation"]


def load(files):
    return [dict(np.load(f)) for f in files if os.path.exists(f)]


def queries(d):
    """(k, idx, truth (len(idx), H, 3), valid (len(idx), H)) for each query time of one seed."""
    t, S, St = d["t"], d["states"], d["status"]
    for q in QUERY_T:
        k = int(np.argmin(np.abs(t - q)))
        idx = np.flatnonzero((St[k] == 1) & (S[k, :, 2] > ELEV + 1.0))
        if not len(idx):
            continue
        ks = np.minimum(k + np.round(HORIZONS / 0.1).astype(int), len(t) - 1)
        truth = S[ks][:, idx, :3].transpose(1, 0, 2)
        valid = (St[ks][:, idx] == 1).T & (k + np.round(HORIZONS / 0.1).astype(int) < len(t))[None]
        yield k, idx, truth, valid


def evaluate(pred, files=TEST):
    err = [[] for _ in HORIZONS]
    for d in load(files):
        for k, idx, truth, valid in queries(d):
            p = pred.predict(d["states"][: k + 1], d["status"][: k + 1], d["t"][: k + 1], idx, HORIZONS)
            e = np.linalg.norm(p - truth, axis=-1)
            for h in range(len(HORIZONS)):
                err[h].extend(e[valid[:, h], h])
    return {float(h): (float(np.mean(e)), float(np.median(e)), float(np.percentile(e, 90)), len(e)) for h, e in zip(HORIZONS, err)}


def report(name, res):
    print(f"{name}: " + "  ".join(f"{h:g}s {m:.2f}/{p50:.2f}/{p90:.2f}" for h, (m, p50, p90, n) in res.items()) + "   (mean/median/p90 m)")


if __name__ == "__main__":
    spec = importlib.util.spec_from_file_location("pred", sys.argv[1]); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    p = mod.Predictor(GIVEN)
    if hasattr(p, "fit"):
        p.fit(load(TRAIN))
    report(os.path.basename(sys.argv[1]), evaluate(p))

"""Train the leg reachability model P(hit | state, planned T) on dev/collect.py labels.

uv run python dev/train_legs.py data/legs_all.npz BalloonPoppingGymEnv/agents/leg_model.npz

ponytail: numpy MLP + Adam, no torch. Inference lives in agents/leg_model.py.
"""

import sys

import numpy as np

from BalloonPoppingGymEnv.agents.leg_model import LegModel, forward  # noqa: E402

H, EPOCHS, BATCH, LR = 64, 60, 512, 2e-3


def init(rng, sizes):
    return [(rng.normal(0, np.sqrt(2 / a), (a, b)), np.zeros(b)) for a, b in zip(sizes[:-1], sizes[1:])]


def grads(params, x, y):
    # forward with cache
    hs, a = [x], x
    for W, b in params[:-1]:
        a = np.maximum(a @ W + b, 0)
        hs.append(a)
    W, b = params[-1]
    logit = (a @ W + b)[:, 0]
    p = 1 / (1 + np.exp(-logit))
    loss = -np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9))
    d = ((p - y) / len(y))[:, None]
    g = []
    for i in range(len(params) - 1, -1, -1):
        W, b = params[i]
        g.append((hs[i].T @ d, d.sum(0)))
        if i:
            d = (d @ W.T) * (hs[i] > 0)
    return loss, g[::-1]


def main(src, dst):
    d = np.load(src)
    X, y = d["X"], d["Y"][:, 0]
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]
    n_val = len(X) // 10
    mu, sd = X[n_val:].mean(0), X[n_val:].std(0) + 1e-6
    Xn = (X - mu) / sd
    params = init(rng, [X.shape[1], H, H, 1])
    m = [(np.zeros_like(W), np.zeros_like(b)) for W, b in params]
    v = [(np.zeros_like(W), np.zeros_like(b)) for W, b in params]
    step = 0
    for ep in range(EPOCHS):
        idx = rng.permutation(np.arange(n_val, len(X)))
        for i in range(0, len(idx), BATCH):
            j = idx[i : i + BATCH]
            _, g = grads(params, Xn[j], y[j])
            step += 1
            for k, ((W, b), (gW, gb)) in enumerate(zip(params, g)):
                mW, mb = m[k]; vW, vb = v[k]
                mW[:] = 0.9 * mW + 0.1 * gW; mb[:] = 0.9 * mb + 0.1 * gb
                vW[:] = 0.999 * vW + 0.001 * gW**2; vb[:] = 0.999 * vb + 0.001 * gb**2
                c = np.sqrt(1 - 0.999**step) / (1 - 0.9**step)
                W -= LR * c * mW / (np.sqrt(vW) + 1e-8)
                b -= LR * c * mb / (np.sqrt(vb) + 1e-8)
        if ep % 10 == 9 or ep == EPOCHS - 1:
            pv = forward(params, Xn[:n_val])
            acc = ((pv > 0.5) == y[:n_val]).mean()
            ll = -np.mean(y[:n_val] * np.log(pv + 1e-9) + (1 - y[:n_val]) * np.log(1 - pv + 1e-9))
            print(f"epoch {ep + 1}  val logloss {ll:.3f}  acc {acc:.3f}  (base rate {1 - y[:n_val].mean():.3f})")
    LegModel.save(dst, params, mu, sd)
    print("saved", dst)


if __name__ == "__main__":
    main(*sys.argv[1:3])

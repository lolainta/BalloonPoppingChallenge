"""Learned reach model for RL2 tokens: per stretch level, P(hit) and time to hit (dev/train_t1sl.py).
Input = rocket globals (6) + token features without the learned slots; numpy MLP."""
import numpy as np

IN_TOK = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13]  # token columns used (skips t_hat, p_hat, alive)


def load_sl(path):
    d = dict(np.load(path))
    n = int(d["n_layers"])
    layers = [(d[f"W{i}"], d[f"b{i}"]) for i in range(n)]
    mu, sd, ns = d["mu"], d["sd"], int(d["n_str"])

    def sl(tok, g):
        z = (np.concatenate([np.broadcast_to(g, (len(tok), 6)), tok[:, IN_TOK]], 1) - mu) / sd
        for W, b in layers[:-1]:
            z = np.maximum(z @ W + b, 0)
        W, b = layers[-1]
        o = z @ W + b
        p = 1 / (1 + np.exp(-o[:, :ns]))
        t = np.maximum(o[:, ns:], 0.0)
        best = np.argmax(p - 0.02 * t, 1)  # best level: likely hit, prefer faster
        r = np.arange(len(tok))
        return t[r, best] / 10.0, p[r, best]

    return sl

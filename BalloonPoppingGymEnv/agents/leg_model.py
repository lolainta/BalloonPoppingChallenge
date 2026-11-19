"""Learned leg reachability: P(hit | rel_p(3), rel_v(3), v(3), z_axis(3), burn_left, planned T).

Weights trained offline by dev/train_legs.py from simulator rollouts; at run time the agent
only feeds it quantities derived from observations. numpy-only so it runs inside the agent.
"""

from pathlib import Path

import numpy as np


def forward(params, x):
    a = x
    for W, b in params[:-1]:
        a = np.maximum(a @ W + b, 0)
    W, b = params[-1]
    return 1 / (1 + np.exp(-(a @ W + b)[..., 0]))


class LegModel:
    def __init__(self, path=Path(__file__).with_name("leg_model.npz")):
        d = np.load(path)
        n = int(d["n_layers"])
        self.params = [(d[f"W{i}"], d[f"b{i}"]) for i in range(n)]
        self.mu, self.sd = d["mu"], d["sd"]

    def __call__(self, x):
        return forward(self.params, (x - self.mu) / self.sd)

    @staticmethod
    def save(path, params, mu, sd):
        out = {"n_layers": len(params), "mu": mu, "sd": sd}
        for i, (W, b) in enumerate(params):
            out[f"W{i}"], out[f"b{i}"] = W, b
        np.savez(path, **out)

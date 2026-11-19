"""Synthetic post-launch rows + their exact log-probs under a live policy (for the fake-worker checks)."""
import numpy as np
import torch

from dev import async_ppo as ap
from dev.rl3_net import Net3, load_npz

G, NT, F = 64, 100, 15


def obs(rng, n):
    X = (0.3 * rng.standard_normal((n, G + NT * F))).astype(np.float32)
    X[:, ap.LAUNCHED_COL], X[:, ap.LAUNCHED_COL + 1] = 1.0, 0.5
    tok = X[:, G:].reshape(n, NT, F); tok[:, :, 14] = 0.0; tok[:, :10, 14] = 1.0; tok[:, :, 8] = 0.0
    return X


def segment(live, rng, width):
    """One 64-row segment of actions of the given width (5: no res head, 8: with it) and the policy's log-probs."""
    X = obs(rng, ap.SEG)
    A = np.zeros((ap.SEG, width)); A[:, 0] = rng.integers(0, 10, ap.SEG); A[:, 1] = rng.integers(0, 4, ap.SEG)
    net = Net3(); load_npz(net, live)
    with torch.no_grad():
        lg, val = net(torch.tensor(X))
        lp = torch.log_softmax(lg[:, :100], -1).gather(1, torch.tensor(A[:, :1]).long())[:, 0] \
            + torch.log_softmax(lg[:, 100:104], -1).gather(1, torch.tensor(A[:, 1:2]).long())[:, 0]
        if width == 8:
            m, ls = lg[:, -3:].numpy(), net.res_logstd.numpy()
            A[:, 5:] = (m + np.exp(ls) * rng.standard_normal((ap.SEG, 3))).astype(np.float32)
            lp = lp + ap.res_gauss(net, torch.tensor(m), torch.tensor(A[:, 5:], dtype=torch.float32), torch.tensor(X))[0]
    return {"obs": X, "act": A, "logp": lp.numpy().astype(np.float64), "val": val.numpy().astype(np.float64),
            "rew": rng.integers(0, 2, ap.SEG).astype(np.float64), "done": np.zeros(ap.SEG, bool)}

"""C: d(HOENT term)/d(current target's logit) = 0 with p_switch detached (as the learner uses it), non-zero with a
deliberately non-detached p_switch. Real rows: target logits of POLICY on ROWS (a leaf tensor, so the alternatives are fixed).
source dev/hoent_check/env.sh; $PY dev/hoent_check/check_c.py POLICY.npz ROWS.npz [...]"""
import os
os.environ.setdefault("RL3_HOENT", "0.3")
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dev import async_ppo as ap  # noqa: E402
from dev.rl3_net import Net3, load_npz  # noqa: E402

assert Path(ap.__file__).resolve().is_relative_to(ROOT), ap.__file__
net = Net3(); load_npz(net, sys.argv[1])
R = [dict(np.load(f)) for f in sys.argv[2:]]
X = torch.tensor(np.concatenate([r["obs"] for r in R]), dtype=torch.float32)
A = torch.tensor(np.concatenate([r["act"] for r in R]), dtype=torch.float32)
with torch.no_grad():
    tl0 = net(X, tgt=A[:, 0].long())[0][:, :100]
tok = X[:, ap.TOK0:].reshape(len(X), -1, ap.TOK_F)
cur = (tok[..., ap.TOK_CUR] > 0.5) & (tok[..., ap.TOK_ALIVE] > 0.5)
for detach in (True, False):
    tl = tl0.clone().requires_grad_(True)
    m, h, w = ap.hoent(tl, X, detach=detach)
    w = w * m
    term = -0.02 * (h * w).sum() / w.sum()  # = the loss term (alpha_h 0.02) the learner adds
    term.backward()
    g = tl.grad
    gc, ga = g[cur & m[:, None]], g[~cur & m[:, None] & (tok[..., ap.TOK_ALIVE] > 0.5)]
    print(f"p_switch {'detached    ' if detach else 'NOT detached'}: rows {int(m.sum())}  d term / d current logit: max |.| {gc.abs().max():.3e}  "
          f"(exactly 0 on all rows: {bool((gc == 0).all())})   d term / d alternative logits: max |.| {ga.abs().max():.3e}")

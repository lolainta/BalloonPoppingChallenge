"""D (synthetic): the learner's RL3_HOENT block alone (loss -= alpha_h * weighted H_cond, a_h_loss = log_alpha_h * (H - target),
Adam, clamp) on the fixed hand-off rows of ROWS, net from POLICY: alpha_h rises while H < target, H rises, alpha_h falls once
H overshoots.  source dev/hoent_check/env.sh; $PY dev/hoent_check/check_d2.py POLICY.npz STEPS ROWS.npz [...]"""
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
torch.manual_seed(0)
net = Net3(); load_npz(net, sys.argv[1])
R = [dict(np.load(f)) for f in sys.argv[3:]]
X = torch.tensor(np.concatenate([r["obs"] for r in R]), dtype=torch.float32)
A = torch.tensor(np.concatenate([r["act"] for r in R]), dtype=torch.float32)
with torch.no_grad():
    keep = ap.hoent(net(X, tgt=A[:, 0].long())[0][:, :100], X)[0]
X, A = X[keep], A[keep]
opt = torch.optim.Adam(net.parameters(), lr=3e-4)
log_alpha_h = torch.tensor(float(np.log(0.02)), requires_grad=True); aopt = torch.optim.Adam([log_alpha_h], lr=3e-3)
for it in range(int(sys.argv[2])):
    m, h, w = ap.hoent(net(X, tgt=A[:, 0].long())[0][:, :100], X)
    w = w * m
    h_ho = (h * w).sum() / w.sum()
    loss = -log_alpha_h.exp().detach() * h_ho
    opt.zero_grad(); loss.backward(); opt.step()
    a_h = log_alpha_h * (h_ho.detach() - ap.HOENT)
    aopt.zero_grad(); a_h.backward(); aopt.step()
    with torch.no_grad():
        log_alpha_h.clamp_(ap.LOG_ALPHA_MIN, ap.LOG_ALPHA_MAX)
    if it % 5 == 0 or it == int(sys.argv[2]) - 1:
        print(f"step {it:3d}  hoent {float(h_ho):.3f}  alpha_h {float(log_alpha_h.exp()):.4f}  mean p_switch {float(w.sum() / m.sum()):.3f}", flush=True)

"""D: the loss quantity (p_switch-weighted H_cond of the hand-off rows) of each exported policy on fixed worker rows.
source dev/hoent_check/env.sh; $PY dev/hoent_check/check_d.py ROWS.npz[,ROWS2.npz..] POLICY.npz [POLICY2.npz ...]"""
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
R = [dict(np.load(f)) for f in sys.argv[1].split(",")]
X = torch.tensor(np.concatenate([r["obs"] for r in R]), dtype=torch.float32)
A = torch.tensor(np.concatenate([r["act"] for r in R]), dtype=torch.float32)
for p in sys.argv[2:]:
    net = Net3(); load_npz(net, p); net.eval()
    with torch.no_grad():
        m, h, ps = ap.hoent(net(X, tgt=A[:, 0].long())[0][:, :100], X)
    w = ps * m
    print(f"{Path(p).parent.name}/{Path(p).name}: hand-off rows {int(m.sum())}  weighted H_cond {float((h * w).sum() / w.sum()):.4f}  "
          f"mean H_cond {float(h[m].mean()):.4f}  mean p_switch {float(ps[m].mean()):.4f}")

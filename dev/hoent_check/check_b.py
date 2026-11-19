"""B: row types and the HOENT quantities (async_ppo.hoent) on real worker rows of the s4res2 policy.
source dev/hoent_check/env.sh; $PY dev/hoent_check/check_b.py POLICY.npz ROWS.npz [ROWS2.npz ...]"""
import os
os.environ.setdefault("RL3_HOENT", "0.3")
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dev import async_ppo as ap, rl3_net  # noqa: E402
from dev.rl3_net import Net3, load_npz  # noqa: E402

for m in (ap, rl3_net):
    assert Path(m.__file__).resolve().is_relative_to(ROOT), m.__file__
net = Net3(); load_npz(net, sys.argv[1]); net.eval()
R = [dict(np.load(f)) for f in sys.argv[2:]]
X = torch.tensor(np.concatenate([r["obs"] for r in R]), dtype=torch.float32)
A = torch.tensor(np.concatenate([r["act"] for r in R]), dtype=torch.float32)
n_eps = sum(int(r["done"].sum()) for r in R)
with torch.no_grad():
    lg, _ = net(X, tgt=A[:, 0].long())
    m, h, ps = ap.hoent(lg[:, :100], X)
    tok = X[:, ap.TOK0:].reshape(len(X), -1, ap.TOK_F)
    cur = (tok[..., ap.TOK_CUR] > 0.5) & (tok[..., ap.TOK_ALIVE] > 0.5)
    post = X[:, ap.LAUNCHED_COL] >= 0.5
    nocur, ho = post & ~cur.any(1), m
    mid = post & cur.any(1) & ~m
    switch = (lg[:, :100].argmax(1) != cur.float().argmax(1)) & ho
m_, h_, p_ = ho.numpy(), h.numpy(), ps.numpy()
print(f"rows {len(X)}  episodes {n_eps}  post-launch {int(post.sum())} ({post.sum() / n_eps:.1f} / episode)  HOENT_T {ap.HOENT_T}")
for nm, k in (("no-current", nocur), ("hand-off", ho), ("mid-chase", mid)):
    print(f"  {nm:10s} {int(k.sum()):5d} rows  {k.sum() / n_eps:5.1f} / episode  {100 * k.sum() / post.sum():5.1f}% of post-launch")
print(f"hand-off rows: argmax switches {100 * switch.sum() / ho.sum():.0f}%  ({switch.sum() / n_eps:.1f} / episode)")
print(f"hand-off rows: mean H_cond {h_[m_].mean():.3f}  median {np.median(h_[m_]):.3f}  mean p_switch {p_[m_].mean():.3f}  "
      f"p_switch-weighted H_cond (the loss quantity) {(h_ * p_ * m_).sum() / (p_ * m_).sum():.3f}")
print(f"mid-chase rows (not in the term): mean H_cond {h_[mid.numpy()].mean():.3f}  mean p_switch {p_[mid.numpy()].mean():.3f}")
np.set_printoptions(precision=3, suppress=True)
print("hand-off H_cond deciles:", np.quantile(h_[m_], np.linspace(0.1, 0.9, 9)))

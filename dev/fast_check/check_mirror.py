"""numpy (rl3_agent.forward: fast_u) vs torch (Net3: the fast mean columns before the residual's) on synthetic post-launch observations,
random non-zero heads (8-output res, tc, tc2, fast + fastc), rows with / without a token flagged "current target"; export -> npz -> numpy;
load_npz(export) round trip.   source dev/fast_check/env.sh; RL3_FAST=10 $PY dev/fast_check/check_mirror.py"""
import os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from dev import rl3_net  # noqa: E402
from BalloonPoppingGymEnv.agents import rl3_agent  # noqa: E402

assert all(Path(m.__file__).resolve().is_relative_to(ROOT) for m in (rl3_net, rl3_agent))
torch.manual_seed(0); rng = np.random.default_rng(0)
net = rl3_net.Net3()
with torch.no_grad():
    for p in net.parameters():
        p.add_(0.1 * torch.randn_like(p))
tmp = Path(os.environ.get("TMPDIR", "/tmp")) / f"mirrorf_{os.getpid()}.npz"; rl3_net.export(net, tmp)
P = dict(np.load(tmp))
net2 = rl3_net.Net3(); rl3_net.load_npz(net2, tmp); tmp.unlink()
rt = max(float((a - b).abs().max()) for a, b in zip(net.state_dict().values(), net2.state_dict().values()) if a.is_floating_point())
ag = rl3_agent.RL3Agent.__new__(rl3_agent.RL3Agent); ag.net = P
B = 32
X = np.zeros((B, rl3_agent.N_OBS), np.float32); X[:, :64] = rng.standard_normal((B, 64)); X[:, 62] = 1.0
tok = X[:, 64:].reshape(B, 100, 15); n = rng.integers(3, 40, B)
cur = np.full(B, -1)
for r in range(B):
    tok[r, :n[r]] = rng.standard_normal((n[r], 15)); tok[r, :n[r], 14] = 1.0; tok[r, :n[r], 8] = 0.0
    tok[r, n[r]: n[r] + 3] = rng.standard_normal((3, 15)); tok[r, n[r]: n[r] + 3, 14] = 0.0; tok[r, n[r]: n[r] + 3, 8] = 0.0
    if r % 4:  # 3 of 4 rows: a current target
        cur[r] = rng.integers(n[r]); tok[r, cur[r], 8] = 1.0
nr = net.res.out_features
net.double()
with torch.no_grad():
    lo, _ = net(torch.tensor(X, dtype=torch.float64))
d_f, d_r, d_c = 0.0, 0.0, 0.0
for r in range(B):
    tl, sl, *_ = ag.forward(X[r]); ag.cond(int(np.argmax(tl)), sl, tl)  # (torch without tgt: tc / tc2 on the argmax target)
    d_f = max(d_f, np.abs(ag.fast_u - lo[r, -2 * nr: -nr].numpy()).max())
    d_r = max(d_r, np.abs(ag.res_u - lo[r, -nr:].numpy()).max())
    if cur[r] >= 0:
        d_c = max(d_c, float(np.abs(ag.h_tok[cur[r]] @ P["fastc.w"] + P["fastc.b"]).max()))
print(f"fast {P['fast.w'].shape} fastc {P['fastc.w'].shape} logstd {P['fast.logstd']} k {P['fast.k']}; export -> load_npz round trip "
      f"max |diff| {rt:.1e};  max |numpy - torch| fast mean {d_f:.2e} (fastc term up to {d_c:.2f} on the {int((cur >= 0).sum())} rows "
      f"with a current target, 0 on the {int((cur < 0).sum())} without), residual mean {d_r:.2e}")
# rows without a current target: the numpy fast mean = fast(pg) only (torch matches it above)
nc = np.flatnonzero(cur < 0)
fc_np = []
for r in nc:
    ag.forward(X[r]); h = ag.h_tok; m = tok[r, :, 14] > 0.5
    pg = np.concatenate([(h * m[:, None]).sum(0) / m.sum(), np.maximum(X[r, :64] @ P["g.w"] + P["g.b"], 0)])
    fc_np.append(np.abs(ag.fast_u - (pg @ P["fast.w"] + P["fast.b"])).max())
print(f"rows without a current target: fast mean = fast(pg) only, max |diff| {max(fc_np):.1e}")

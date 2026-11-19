"""numpy (rl3_agent.forward + cond) vs torch (Net3, tgt) on synthetic post-launch observations, random non-zero tc / tc2 heads, an
8-output residual (RL3_RES3, RL3_TCOND, RL3_TCOND2); rows with 1 / 2 / many live balloons; export -> npz -> numpy; the second
choice (rl3_agent.second vs the torch definition); tc2_off=1 / tc_off=1 / both drop exactly their term.
source dev/tc2_check/env.sh; $PY dev/tc2_check/check_mirror.py"""
import os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from dev import rl3_net  # noqa: E402
from BalloonPoppingGymEnv.agents import rl3_agent  # noqa: E402

assert all(Path(m.__file__).resolve().is_relative_to(ROOT) for m in (rl3_net, rl3_agent))
assert os.environ.get("RL3_TCOND2") and os.environ.get("RL3_RES3")
torch.manual_seed(0); rng = np.random.default_rng(0)
net = rl3_net.Net3()
with torch.no_grad():
    for p in net.parameters():
        p.add_(0.1 * torch.randn_like(p))
tmp = Path(os.environ.get("TMPDIR", "/tmp")) / f"mirror2_{os.getpid()}.npz"; rl3_net.export(net, tmp)
ag = rl3_agent.RL3Agent.__new__(rl3_agent.RL3Agent); ag.net = dict(np.load(tmp)); tmp.unlink()
B = 32
X = np.zeros((B, rl3_agent.N_OBS), np.float32); X[:, :64] = rng.standard_normal((B, 64)); X[:, 62] = 1.0
tok = X[:, 64:].reshape(B, 100, 15); n = rng.integers(3, 40, B); n[:3] = (1, 2, 2)
for r in range(B):
    tok[r, :n[r]] = rng.standard_normal((n[r], 15)); tok[r, :n[r], 14] = 1.0
    tok[r, n[r]: n[r] + 3] = rng.standard_normal((3, 15)); tok[r, n[r]: n[r] + 3, 14] = 0.0  # dead balloons after the live ones
tgt = np.array([rng.integers(n[r]) for r in range(B)])
net.double()
with torch.no_grad():
    lo, _ = net(torch.tensor(X, dtype=torch.float64), tgt=torch.tensor(tgt))
    lo0, _ = net(torch.tensor(X, dtype=torch.float64))  # tgt None: the argmax target
    t2 = net.tc2.weight.clone(); net.tc2.weight.zero_(); b2 = net.tc2.bias.clone(); net.tc2.bias.zero_()
    lz, _ = net(torch.tensor(X, dtype=torch.float64), tgt=torch.tensor(tgt))
    net.tc2.weight.copy_(t2); net.tc2.bias.copy_(b2)
nr = net.res.out_features
ds, dr, ds0, n2, none = 0.0, 0.0, 0.0, 0, 0
for r in range(B):
    tl, sl, *_ = ag.forward(X[r]); s = ag.cond(int(tgt[r]), sl, tl)
    ds = max(ds, np.abs(s - lo[r, 100:104].numpy()).max()); dr = max(dr, np.abs(ag.res_u - lo[r, -nr:].numpy()).max())
    j = ag.j2
    z = tl.astype(np.float64).copy(); z[tgt[r]] = -np.inf; z[n[r]:] = -np.inf  # reference: the live ones other than the target
    assert j == (None if n[r] < 2 else int(np.argmax(z))), (r, j)
    none += j is None; n2 += j is not None
    tl, sl, *_ = ag.forward(X[r]); s = ag.cond(int(np.argmax(tl)), sl, tl)
    ds0 = max(ds0, np.abs(s - lo0[r, 100:104].numpy()).max(), np.abs(ag.res_u - lo0[r, -nr:].numpy()).max())
print(f"residual outputs {nr}, tc {tuple(net.tc.weight.shape)}, tc2 {tuple(net.tc2.weight.shape)}; second choice = reference "
      f"on {B} rows ({none} with < 2 live: none);  max |numpy - torch| stretch logits {ds:.2e}, residual mean {dr:.2e}, "
      f"argmax-target path {ds0:.2e}")
# the < 2 live row: torch's tc2 term is 0 there (= the zero-tc2 net); rows with a second choice: it is not
print("torch tc2 term on the 1-live row: %.1e;  on the others min %.2e" % (
    float((lo[0] - lz[0]).abs().max()), float((lo[1:] - lz[1:]).abs()[:, 100:].max(1).values.min())))
r = 5
for kw in ({"tc2_off": 1}, {"tc_off": 1}, {"tc_off": 1, "tc2_off": 1}):
    ag.tc_off, ag.tc2_off = kw.get("tc_off", 0), kw.get("tc2_off", 0)
    tl, sl0, *_ = ag.forward(X[r]); u0 = ag.res_u.copy(); s = ag.cond(int(tgt[r]), sl0, tl); u = ag.res_u
    exp_s, exp_u = sl0.copy(), u0.copy()
    for name, j in (("tc", None if ag.tc_off else int(tgt[r])), ("tc2", None if ag.tc2_off else rl3_agent.RL3Agent.second(tl, int(tgt[r])))):
        if j is not None:
            k = 9; w, b = ag.net[f"{name}.w"], ag.net[f"{name}.b"]
            c = np.concatenate([ag.h_tok[j] @ w[:, :k] + b[:k], ag.h_tok[j] @ w[:, k:] + b[k:]])
            exp_s, exp_u = exp_s + c[:4], exp_u + c[4:]
    print(kw, "-> exactly the remaining terms:", np.array_equal(s, exp_s) and np.array_equal(u, exp_u),
          " unconditioned:", np.array_equal(s, sl0) and np.array_equal(u, u0))

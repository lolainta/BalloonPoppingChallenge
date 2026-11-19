"""numpy (rl3_agent.forward + cond) vs torch (Net3, tgt) on synthetic post-launch observations, a random non-zero tc head, for
the residual widths 0 / 3 / 5 (RL3_RES / RL3_RES2 unset or set; RL3_TCOND=1); export -> npz -> numpy.
python dev/tcond_check/check_mirror.py"""
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if len(sys.argv) == 1:
    for env in ({}, {"RL3_RES": "0.25"}, {"RL3_RES": "0.25", "RL3_RES2": "1"}):
        e = {k: v for k, v in os.environ.items() if not k.startswith("RL3_RES")} | env | {"RL3_TCOND": "1", "RL3_ATTC": "1"}
        subprocess.run([sys.executable, __file__, "x"], env=e, check=True)
    sys.exit()
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
tmp = Path(os.environ.get("TMPDIR", "/tmp")) / f"mirror_{os.getpid()}.npz"; rl3_net.export(net, tmp)
ag = rl3_agent.RL3Agent.__new__(rl3_agent.RL3Agent); ag.net = dict(np.load(tmp)); tmp.unlink()
X = np.zeros((16, rl3_agent.N_OBS), np.float32); X[:, :64] = rng.standard_normal((16, 64)); X[:, 62] = 1.0
tok = X[:, 64:].reshape(16, 100, 15); n = rng.integers(3, 40, 16)
for r in range(16):
    tok[r, :n[r]] = rng.standard_normal((n[r], 15)); tok[r, :n[r], 14] = 1.0
tgt = np.array([rng.integers(n[r]) for r in range(16)])
net.double()
with torch.no_grad():
    lo, _ = net(torch.tensor(X, dtype=torch.float64), tgt=torch.tensor(tgt))
nr = net.res.out_features if net.res is not None else 0
ds, dr = 0.0, 0.0
for r in range(16):
    tl, sl, *_ = ag.forward(X[r]); sl = ag.cond(int(tgt[r]), sl)
    ds = max(ds, np.abs(sl - lo[r, 100:104].numpy()).max())
    if nr:
        dr = max(dr, np.abs(ag.res_u - lo[r, -nr:].numpy()).max())
ag.tc_off = 1; tl, sl0, *_ = ag.forward(X[0])
print(f"residual outputs {nr}, tc {tuple(net.tc.weight.shape)}: max |numpy - torch| stretch logits {ds:.2e}, residual mean {dr:.2e};"
      f" tc_off=1 leaves them unconditioned: {np.array_equal(ag.cond(int(tgt[0]), sl0), sl0)}")

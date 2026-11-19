"""Defect-2: GPU time of one learner update (EPOCHS x 4 minibatches of async_ppo.flight_loss + SIL + backward + clip + Adam, the learner's
loop) for --batch FULL rows, with RL3_FAST (+ FAST rows at the share SHARE, two SIL batches) or without.
source dev/fast_check/env.sh; CUDA_VISIBLE_DEVICES=0 [RL3_FAST=10] RL3_NEWENT=0.35 .venv-gpu/bin/python dev/fast_check/time_update.py 28800 [SHARE]"""
import sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dev import async_ppo as ap, rl3_net  # noqa: E402
from dev.res_check.fakeseg import obs  # noqa: E402

assert Path(ap.__file__).resolve().is_relative_to(ROOT)
dev = "cuda"
nf = int(sys.argv[1]); share = float(sys.argv[2]) if len(sys.argv) > 2 else 0.79
nq = int(round(nf * share / (1 - share))) if ap.FAST else 0
rng = np.random.default_rng(0)
n = nf + nq
X = obs(rng, n)
tok = X[:, 64:].reshape(n, 100, 15); tok[:, :30, 14] = 1.0  # 30 live balloons
aw = 3 + 2 + 8 + (1 if ap.FAST else 0)
A = np.zeros((n, aw), np.float32); A[:, 0] = rng.integers(0, 30, n); A[:, 1] = rng.integers(0, 4, n)
A[:, -8:] = 0.1 * rng.standard_normal((n, 8))
if ap.FAST:
    A[:nf, -9] = 1.0; tok[nf:, 0, 8] = 1.0
net = rl3_net.Net3().to(dev)
opt = torch.optim.Adam(net.parameters(), lr=1e-4)
al = tuple(torch.tensor(float(np.log(0.02)), device=dev, requires_grad=True) for _ in range(3))
tg = lambda A_: {"tgt": A_[:, 0].long()}  # noqa: E731
T = lambda z: torch.tensor(z, device=dev)  # noqa: E731
O, ACT = T(X), T(A)
OLP, ADV, RET = (T(rng.standard_normal(n).astype(np.float32)) for _ in range(3))
Xd, Ad, Rd = O[:ap.SIL_BATCH], ACT[:ap.SIL_BATCH], RET[:ap.SIL_BATCH]
Xq, Aq, Rq = O[-ap.SIL_BATCH:], ACT[-ap.SIL_BATCH:], RET[-ap.SIL_BATCH:]
sizes = ap.dims(True)


def update():
    fi, qi = torch.arange(nf, device=dev), torch.arange(nf, n, device=dev)
    for _ in range(ap.EPOCHS):
        fch = fi[torch.randperm(nf, device=dev)].split(nf // 4)
        for f_, q_ in zip(fch, torch.tensor_split(qi[torch.randperm(nq, device=dev)], len(fch))):  # (= async_ppo.learn)
            opt.zero_grad(); nrm = (len(f_), max(len(q_), 1), len(f_) + len(q_)) if ap.FAST else None
            for p in q_.split(len(f_)) if len(q_) else ():
                ap.flight_loss(net, O[p], ACT[p], OLP[p], ADV[p], RET[p], sizes, al, tg, True, nrm)[0].backward()
            loss = ap.flight_loss(net, O[f_], ACT[f_], OLP[f_], ADV[f_], RET[f_], sizes, al, tg, True, nrm)[0]
            loss = loss + ap.SIL_W * ap.sil_loss(net, Xd, Ad, Rd, sizes, tg)
            if ap.FAST:
                loss = loss + ap.SIL_W * ap.sil_loss(net, Xq, Aq, Rq, sizes, tg, fast=True)
            loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), ap.MAXGRAD); opt.step()
    torch.cuda.synchronize()


update()  # warm-up
ts = []
for _ in range(3):
    t0 = time.time(); update(); ts.append(time.time() - t0)
print(f"{torch.cuda.get_device_name()}: RL3_FAST {ap.FAST}: FULL rows {nf} + FAST rows {nq} = {n} rows per update, minibatch ~{n // 4}: "
      f"{ap.EPOCHS * 4} optimizer steps, update {np.mean(ts):.2f} s (runs {', '.join(f'{t:.2f}' for t in ts)}), peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")

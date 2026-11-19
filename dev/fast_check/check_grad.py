"""Defect-1 check: on one fixed batch (synthetic pad / FULL rows, ~79% FAST rows, live settings: ATTC, 8-output res, tc, tc2,
NEWENT, SIL), the gradient of today's heads' parameters (tgt / prior / str / lau / attc / res / tc / tc2) from an RL3_FAST learner's
loss (async_ppo.flight_loss + sil_loss on FULL demo rows + the fast SIL) equals the gradient a non-RL3_FAST learner (the same module
imported without RL3_FAST) computes on the FULL / pad rows alone; the fast head gets gradient only from the FAST rows.
RL3_FAST_DIMS=n / RL3_FAST_STD=s (from the caller): an n-output fast head (FAST rows: deltas in the first n u columns, the rest 0).
source dev/fast_check/env.sh; RL3_NEWENT=0.35 [RL3_FAST_DIMS=3 RL3_FAST_STD=0.03] $PY dev/fast_check/check_grad.py"""
import importlib.util, os, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dev import rl3_net  # noqa: E402
from dev.res_check.fakeseg import obs  # noqa: E402

assert Path(rl3_net.__file__).resolve().is_relative_to(ROOT)
os.environ.pop("RL3_FAST", None)


def load(name, fast):
    if fast:
        os.environ["RL3_FAST"] = "10"
    else:
        os.environ.pop("RL3_FAST", None)
    spec = importlib.util.spec_from_file_location(name, ROOT / "dev/async_ppo.py"); m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ap0, apf = load("ap_off", False), load("ap_fast", True)
assert apf.FAST == 10 and not ap0.FAST and ap0.NEWENT == 0.35 and ap0.NR == 8 and ap0.ATTC
os.environ.pop("RL3_FAST", None); torch.manual_seed(0); net0 = rl3_net.Net3()
with torch.no_grad():
    for p in net0.parameters():
        p.add_(0.05 * torch.randn_like(p))
os.environ["RL3_FAST"] = "10"; netf = rl3_net.Net3(); NFD = netf.fast.out_features
assert apf.NF == NFD and not ap0.NF
print(f"fast head {NFD} outputs, init std {float(netf.fast_logstd.detach().exp()[0]):.4f}, log std clamp lower bound {netf.fast_lsmin:.4f}")
r = netf.load_state_dict(net0.state_dict(), strict=False); assert not r.unexpected_keys and all(k.startswith("fast") for k in r.missing_keys)
with torch.no_grad():
    for n_, p in netf.named_parameters():
        if n_.startswith("fast") and n_ != "fast_logstd":
            p.copy_(0.05 * torch.randn_like(p))
rng = np.random.default_rng(0)


def batch(n, n_pad, frac_fast):
    """Synthetic rows: n_pad pad rows, then post-launch rows (a share frac_fast FAST: current target flagged, flag 0, u = delta)."""
    X = obs(rng, n); X[:n_pad, ap0.LAUNCHED_COL] = 0.0
    fast = np.zeros(n, bool); fast[n_pad:] = rng.random(n - n_pad) < frac_fast
    A = np.zeros((n, 14)); A[:, 0] = rng.integers(0, 10, n); A[:, 1] = rng.integers(0, 4, n); A[:, 2] = (np.arange(n) < n_pad) * rng.integers(0, 2, n)
    A[:n_pad, 3:5] = rng.standard_normal((n_pad, 2)); A[n_pad:, 0] = np.where(fast[n_pad:], 0, A[n_pad:, 0]); A[:n_pad, 0] = 0; A[:n_pad, 1] = 3
    A[:, 5] = (~fast).astype(float); A[:, 6:] = (0.2 * rng.standard_normal((n, 8))).astype(np.float32); A[np.ix_(fast, range(6 + NFD, 14))] = 0
    tok = X[:, 64:].reshape(n, 100, 15)
    tok[fast, rng.integers(0, 10, n)[fast], 8] = 1.0  # FAST rows have a current target
    return X, A, fast


def grads(net, loss):
    net.zero_grad(); loss.backward()
    return {n_: (p.grad.clone() if p.grad is not None else torch.zeros_like(p)) for n_, p in net.named_parameters()}


OLD = ("tgt.", "prior", "str.", "lau.", "attc", "res", "tc.", "tc2.")
for dt in (torch.float64, torch.float32):
    net0.to(dt); netf.to(dt)
    X, A, fast = batch(512, 40, 0.79)
    Xd, Ad, fd = batch(1024, 40, 0.79)  # SIL demo rows
    T = lambda z: torch.tensor(z, dtype=dt)  # noqa: E731
    O, ACT = T(X), T(A)
    OLP = T(rng.standard_normal(len(X)) - 8.0); ADV = T(rng.standard_normal(len(X))); RET = T(rng.standard_normal(len(X)))
    full = ~fast
    for m in (full, fast):  # (the learner normalises per kind)
        ADV[m] = (ADV[m] - ADV[m].mean()) / (ADV[m].std() + 1e-8)
    al = tuple(torch.tensor(float(np.log(0.02)), dtype=dt, requires_grad=True) for _ in range(3))
    tg = lambda A_: {"tgt": A_[:, 0].long()}  # noqa: E731
    noflag = lambda Z: torch.cat([Z[:, :5], Z[:, 6:]], 1)  # noqa: E731  (a non-FAST learner's actions)
    sizes = ap0.dims(True)
    Rd = T(rng.uniform(0, 8, len(Xd)))
    f0 = torch.tensor(full)
    l0 = ap0.flight_loss(net0, O[f0], noflag(ACT[f0]), OLP[f0], ADV[f0], RET[f0], sizes, al, tg)[0] \
        + ap0.SIL_W * ap0.sil_loss(net0, T(Xd)[~torch.tensor(fd)], noflag(T(Ad))[~torch.tensor(fd)], Rd[~torch.tensor(fd)], sizes, tg)
    lf = apf.flight_loss(netf, O, ACT, OLP, ADV, RET, sizes, al, tg)[0] \
        + apf.SIL_W * apf.sil_loss(netf, T(Xd)[~torch.tensor(fd)], T(Ad)[~torch.tensor(fd)], Rd[~torch.tensor(fd)], sizes, tg) \
        + apf.FAST_W * apf.SIL_W * apf.sil_loss(netf, T(Xd)[torch.tensor(fd)], T(Ad)[torch.tensor(fd)], Rd[torch.tensor(fd)], sizes, tg, fast=True)
    g0, gf = grads(net0, l0), grads(netf, lf)
    old = [k for k in g0 if k.startswith(OLD)]
    trunk = [k for k in g0 if not k.startswith(OLD + ("v0.", "v1."))]
    print(f"{dt}: batch {len(X)} rows (pad 40, FULL {int(full.sum()) - 40}, FAST {int(fast.sum())}); grad of the FAST learner vs the non-FAST "
          f"learner on the FULL / pad rows:")
    for name, ks in (("today's heads", old), ("shared trunk (g, t0, t1, attention blocks)", trunk), ("value head (v0, v1)", ["v0.weight", "v1.weight"])):
        da = max(float((g0[k] - gf[k]).abs().max()) for k in ks)
        nrm = {k: float(g0[k].norm()) for k in ks}
        rel = max(float((g0[k] - gf[k]).norm()) / max(nrm[k], 1e-30) for k in ks if nrm[k] > 1e-6 * max(nrm.values()))
        print(f"    {name} ({len(ks)} tensors, grad norms {min(nrm.values()):.2e} .. {max(nrm.values()):.2e}): max |diff| {da:.2e}, max "
              f"relative (tensors with norm > 1e-6 of the largest) {rel:.2e}")
    print("    fast head grad norms " + ", ".join(f"{k} {float(gf[k].norm()):.2e}" for k in ("fast.weight", "fastc.weight", "fast_logstd")))
    gq = grads(netf, apf.flight_loss(netf, O[f0], ACT[f0], OLP[f0], ADV[f0], RET[f0], sizes, al, tg)[0])
    print("    the same loss on the FULL / pad rows only: fast head grad max |.| " + ", ".join(
        f"{k} {float(gq[k].abs().max()):.1e}" for k in ("fast.weight", "fast.bias", "fastc.weight", "fastc.bias", "fast_logstd")))
    # the learner's gradient accumulation: FAST rows in pieces of the FULL piece's size, denominators of the whole minibatch
    fi, qi = torch.nonzero(f0)[:, 0], torch.nonzero(~f0)[:, 0]
    nrm = (len(fi), len(qi), len(X)); netf.zero_grad()
    for p in qi.split(len(fi)):
        apf.flight_loss(netf, O[p], ACT[p], OLP[p], ADV[p], RET[p], sizes, al, tg, True, nrm, fast_rows=True)[0].backward()
    apf.flight_loss(netf, O[fi], ACT[fi], OLP[fi], ADV[fi], RET[fi], sizes, al, tg, True, nrm)[0].backward()
    ga = {n_: p.grad.clone() for n_, p in netf.named_parameters() if p.grad is not None}
    gw = grads(netf, apf.flight_loss(netf, O, ACT, OLP, ADV, RET, sizes, al, tg)[0])
    print(f"    accumulated over {1 + len(qi.split(len(fi)))} pieces vs the whole minibatch at once: max |grad diff| over all parameters "
          f"{max(float((ga[k] - gw[k]).abs().max()) for k in ga):.2e} (largest grad {max(float(g.abs().max()) for g in gw.values()):.2e}; "
          f"fast head {max(float((ga[k] - gw[k]).abs().max()) for k in ga if k.startswith('fast')):.2e} of its largest grad "
          f"{max(float(gw[k].abs().max()) for k in ga if k.startswith('fast')):.2e}, the rest {max(float((ga[k] - gw[k]).abs().max()) for k in ga if not k.startswith('fast')):.2e} of "
          f"{max(float(gw[k].abs().max()) for k in ga if not k.startswith('fast')):.2e})")
    # the fast head's gradient comes from the FAST rows only: the same loss with the FAST rows' ADV / deltas changed -> unchanged old-head grads
    ACT2, ADV2 = ACT.clone(), ADV.clone(); ACT2[torch.tensor(fast), 6: 6 + NFD] += 0.3; ADV2[torch.tensor(fast)] *= -1
    gf2 = grads(netf, apf.flight_loss(netf, O, ACT2, OLP, ADV2, RET, sizes, al, tg)[0])
    gf1 = grads(netf, apf.flight_loss(netf, O, ACT, OLP, ADV, RET, sizes, al, tg)[0])
    print(f"    FAST rows' deltas / advantages changed: today's heads' grads max change {max(float((gf1[k] - gf2[k]).abs().max()) for k in old):.1e}, "
          f"fast.weight grad change {float((gf1['fast.weight'] - gf2['fast.weight']).abs().max()):.2e}")

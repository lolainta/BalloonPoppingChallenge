"""Function-preserving widening of an exported RL3 policy (Net2Net-style, DECISION_LOG §315).

Every residual-stream unit is duplicated k times (D -> kD; every attention head -> k heads of the same size), the in-weights of a
duplicated input are divided by k, and LayerNorm over the k-fold duplicated vector equals LayerNorm over the original: the
deployed numpy forward (agents/rl3_agent.py) returns the same outputs (checked here on real observations, --obs). Exact copies
would stay copies under training (identical gradients), so small noise on the trunk weights breaks the symmetry (--noise, in
units of each matrix's RMS; checked the same way). Extra layers (RL3_LAYERS > the file's n_layers) start as identities
(rl2_net.zero_blocks). Writes <dst>/async_last.pt for dev/async_ppo.py (its param groups, fresh Adam, steps / version kept)
and <dst>/rl3_policy_<steps>.npz (+ the SL copy next to the source, as the learner exports them).

  RL3_D=128 RL3_HEADS=8 RL3_LAYERS=6 <the learner's other RL3_* env> .venv-gpu/bin/python dev/widen.py \
      --src runs/s40l/rl3_policy_560821153.npz --dst runs/s41a --version 20352 --obs obs40.npz --noise 0.01
"""
import argparse, os, shutil, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TRUNK = ("g.w", "t0.w", "t0x.w", "t1.w")  # (+ every layer's in / out / ff1 / ff2 .w): the matrices that get the symmetry-breaking noise


def widen_npz(P, k):
    """The exported weight dict P (matrices stored (in, out)) widened k-fold; function-preserving."""
    D, H = P["g.w"].shape[1], int(P["heads"])
    out = lambda W: np.concatenate([W] * k, 1)  # noqa: E731  duplicated output units
    inn = lambda W: np.concatenate([W] * k, 0) / k  # noqa: E731  duplicated input units: each copy gets 1/k of the weight
    vec = lambda b: np.concatenate([b] * k)  # noqa: E731
    pg = lambda W: np.concatenate([W[:D]] * k + [W[D:]] * k, 0) / k  # noqa: E731  input = [pooled(kD), ge(kD)] (rl3_agent.forward)
    Q = {}
    for key, v in P.items():
        name = layer_name(key)
        if name in ("g.w", "t0.w", "t0x.w"):
            Q[key] = out(v)
        elif name in ("g.b", "t0.b", "t1.b", "v0.b", "out.b", "ff1.b", "ff2.b", "ln1.w", "ln1.b", "ln2.w", "ln2.b"):
            Q[key] = vec(v)
        elif name in ("t1.w", "out.w", "ff1.w", "ff2.w"):
            Q[key] = out(inn(v))
        elif name == "in.w":  # (D, 3D): q | k | v blocks, each duplicated on its own -> heads 0..H-1 then their copies
            w = inn(v)
            Q[key] = np.concatenate([out(w[:, j * D:(j + 1) * D]) for j in range(3)], 1)
        elif name == "in.b":
            Q[key] = np.concatenate([vec(v[j * D:(j + 1) * D]) for j in range(3)])
        elif name in ("tgt.w", "v1.w", "tc.w", "tc2.w", "fastc.w"):  # read a token embedding h (or v0's output)
            Q[key] = inn(v)
        elif name == "v0.w":
            Q[key] = out(pg(v))
        elif name in ("str.w", "lau.w", "attc.w", "att.w", "res.w", "fast.w"):  # read pg
            Q[key] = pg(v)
        elif key == "heads":
            Q[key] = np.int64(H * k)
        else:  # biases of the unchanged-size heads, prior / lprior, log stds, bounds, n_layers, t_min, gnc.*, feat.chain
            Q[key] = v
    return Q


def layer_name(key):
    """'l3.ff1.w' -> 'ff1.w' (a block's weight), anything else unchanged ('lau.w', 'lprior' included)."""
    return key.split(".", 1)[1] if key[:1] == "l" and key[1:].split(".")[0].isdigit() else key


def add_noise(Q, noise, rng):
    for key, v in Q.items():
        if key in TRUNK or layer_name(key) in ("in.w", "out.w", "ff1.w", "ff2.w"):
            Q[key] = (v + rng.normal(0, noise * np.sqrt((v ** 2).mean()), v.shape)).astype(v.dtype)
    return Q


def numpy_agent(P):
    from BalloonPoppingGymEnv.agents.rl3_agent import RL3Agent
    a = RL3Agent.__new__(RL3Agent)  # (as the training workers: no __init__, just the weights)
    a.net = P
    return a


def compare(P, Q, X, label):
    """Deployed numpy forward of P vs Q on observation rows X: max abs differences and the target argmax agreement."""
    a, b = numpy_agent(P), numpy_agent(Q)
    d = {k: 0.0 for k in ("tl", "sl", "ll", "val", "al", "res", "fast", "sl_cond", "res_cond")}
    agree = n_post = 0
    t = [0.0, 0.0]
    for x in X:
        outs = []
        for i, ag in enumerate((a, b)):
            t0 = time.perf_counter(); tl, sl, ll, val, al = ag.forward(x); t[i] += time.perf_counter() - t0
            post = x[62] > 0.5 and (tl > -5e8).sum() > 1
            slc = ag.cond(int(np.argmax(tl)), sl, tl) if post else sl
            outs.append((tl, sl, ll, val, al, ag.res_u, getattr(ag, "fast_u", np.zeros(1)), slc, ag.res_u.copy(), post))
        (tl, sl, ll, val, al, res, fast, slc, resc, post), (tl2, sl2, ll2, val2, al2, res2, fast2, slc2, resc2, _) = outs
        live = tl > -5e8
        for key, u, v in (("tl", tl[live], tl2[live]), ("sl", sl, sl2), ("ll", ll, ll2), ("val", val, val2), ("al", al, al2),
                          ("res", res, res2), ("fast", fast, fast2), ("sl_cond", slc, slc2), ("res_cond", resc, resc2)):
            d[key] = max(d[key], float(np.abs(np.asarray(u) - np.asarray(v)).max()))
        if post:
            n_post += 1; agree += int(np.argmax(tl) == np.argmax(tl2))
    print(f"[{label}] {len(X)} rows ({n_post} post-launch): max|diff| " + " ".join(f"{k} {v:.2e}" for k, v in d.items())
          + f" | target argmax agreement {agree}/{n_post} | numpy forward {1e3 * t[0] / len(X):.2f} -> {1e3 * t[1] / len(X):.2f} ms/row", flush=True)
    return d, agree, n_post


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="exported rl3_policy_<steps>.npz (its rl2_sl_<steps>.npz next to it)")
    ap.add_argument("--dst", required=True, help="run directory to create (async_last.pt, rl3_policy_<steps>.npz, rl2_sl_<steps>.npz)")
    ap.add_argument("--version", type=int, required=True, help="the learner version of --src (progress / log)")
    ap.add_argument("--obs", help="npz with X = observation rows for the parity check (scratchpad/capture_obs.py)")
    ap.add_argument("--noise", type=float, default=0.01, help="symmetry-breaking noise, in units of each trunk matrix's RMS")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_alpha", type=float, default=float(np.log(1e-4)), help="RL3_NEWENT temperature to carry (s40l: at the 1e-4 floor)")
    a = ap.parse_args()
    import torch
    from dev import rl2_net
    from dev.rl3_net import Net3, export, load_npz
    P = dict(np.load(a.src))
    D0, H0, L0 = P["g.w"].shape[1], int(P["heads"]), int(P["n_layers"])
    assert rl2_net.D % D0 == 0 and rl2_net.HEADS == H0 * (rl2_net.D // D0), f"RL3_D={rl2_net.D} RL3_HEADS={rl2_net.HEADS} vs file D {D0} heads {H0}"
    assert rl2_net.LAYERS >= L0, f"RL3_LAYERS={rl2_net.LAYERS} < file n_layers {L0}"
    k = rl2_net.D // D0
    steps = int(Path(a.src).stem.rsplit("_", 1)[1])
    Q = widen_npz(P, k)
    X = np.load(a.obs)["X"] if a.obs else None
    if X is not None:
        compare(P, Q, X, f"exact x{k}")
    if a.noise:
        add_noise(Q, a.noise, np.random.default_rng(a.seed))
        if X is not None:
            compare(P, Q, X, f"noise {a.noise}")
    dst = Path(a.dst); dst.mkdir(parents=True, exist_ok=True)
    pol = dst / f"rl3_policy_{steps}.npz"
    np.savez(pol, **Q)
    sl = Path(a.src).with_name(f"rl2_sl_{steps}.npz")
    if sl.exists():
        shutil.copy2(sl, dst / sl.name)
    # torch side: the learner's Net3 under the same env, loaded from the widened file (extra layers: identities), round trip checked
    net = Net3()
    load_npz(net, pol)
    tmp = dst / ".roundtrip.npz"
    export(net, tmp)
    R = dict(np.load(tmp)); tmp.unlink()
    worst = max(float(np.abs(R[key] - Q[key]).max()) for key in Q if key in R and np.shape(R[key]) == np.shape(Q[key]) and Q[key].dtype.kind == "f")
    assert int(R["n_layers"]) == rl2_net.LAYERS and int(R["heads"]) == rl2_net.HEADS, (R["n_layers"], R["heads"])
    print(f"torch round trip: n_layers {R['n_layers']} heads {R['heads']} max|diff| {worst:.2e}", flush=True)
    if X is not None:  # torch forward vs numpy forward (target logits post-launch, value)
        b = numpy_agent(Q)
        with torch.no_grad():
            lo, val = net(torch.tensor(X))
        dt = dv = 0.0
        for i, x in enumerate(X):
            tl, _, _, v, _ = b.forward(x)
            live = tl > -5e8
            if x[62] > 0.5:
                dt = max(dt, float(np.abs(lo[i, :len(tl)].numpy()[live] - tl[live]).max()))
            dv = max(dv, abs(float(val[i]) - float(v)))
        print(f"torch vs numpy on the widened net: max|diff| target logits {dt:.2e} value {dv:.2e}", flush=True)
    # the learner's optimizer groups (dev/async_ppo.py learn): base, then res, tc, tc2, fast, deep (blocks >= 2), chain
    named = list(net.named_parameters())
    deep = lambda n: n.startswith("blocks.") and int(n.split(".")[1]) >= 2  # noqa: E731
    groups = [[p for n, p in named if not n.startswith(("res", "tc.", "tc2.", "fast", "t0x.")) and not deep(n)]]
    for pre in ("res", "tc.", "tc2.", "fast"):
        groups.append([p for n, p in named if n.startswith(pre)])
    groups.append([p for n, p in named if deep(n)])
    groups.append([p for n, p in named if n.startswith("t0x.")])
    opt = torch.optim.Adam([{"params": g} for g in groups if g], lr=1e-4)
    torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "steps": steps, "version": a.version,
                "log_alpha": a.log_alpha if os.environ.get("RL3_NEWENT") else None, "log_alpha_s": None, "log_alpha_h": None},
               dst / "async_last.pt")
    print(f"{dst}/async_last.pt: steps {steps} version {a.version}, param groups " + " ".join(str(len(g)) for g in groups if g)
          + f", {sum(p.numel() for p in net.parameters()):,} parameters (source {sum(v.size for key, v in P.items() if key.endswith(('.w', '.b'))):,})", flush=True)


if __name__ == "__main__":
    main()

"""Independent launch-value model (user 2026-09-26): predicts an episode's pops from the state at launch and the
launch attitude, trained on the LAUNCHDATA logs (async_ppo workers). Compares the policy's attitude with the best
one the model predicts on held-out launches (random-attitude episodes give the off-policy coverage).
.venv/bin/python dev/launch_value.py ~/bpc_store/launchdata_s4 [out.pt]"""
import glob, sys
import numpy as np
import torch
from torch import nn

G, F, NM = 64, 15, 100  # rl3_agent.G_DIM, rl2_agent.F, N_MAX


def feats(obs, u):
    """Global features + token summary (mean / max over alive balloons) + attitude (inclination, heading offset)."""
    tok = obs[:, G:].reshape(len(obs), NM, F); alive = tok[:, :, 14:15]
    n = alive.sum(1).clip(min=1)
    mean = (tok * alive).sum(1) / n
    mx = np.where(alive > 0, tok, -9.0).max(1)
    incl = 55.0 + 35.0 / (1.0 + np.exp(-u[:, 0])); off = 180.0 * np.tanh(u[:, 1])
    return np.hstack([obs[:, :G], mean, mx, ((incl - 75.0) / 10.0)[:, None], (off / 90.0)[:, None]]).astype(np.float32)


def load(d):
    fs = sorted(glob.glob(f"{d}/*.npz")); O, U, R, RA, FI = [], [], [], [], []
    for i, f in enumerate(fs):
        z = np.load(f); O.append(z["obs"]); U.append(z["act"][:, 3:5]); R.append(z["R"]); RA.append(z["rand_att"])
        FI.append(np.full(len(z["R"]), i))
    return np.vstack(O), np.vstack(U), np.concatenate(R), np.concatenate(RA), np.concatenate(FI)


def main(d, out=None):
    O, U, R, RA, FI = load(d)
    X = feats(O, U); mu, sd = X.mean(0), X.std(0) + 1e-6
    rng = np.random.default_rng(0); va = rng.random(FI.max() + 1)[FI] < 0.15  # split by file (a file = one worker)
    Xt = torch.tensor((X - mu) / sd); Rt = torch.tensor(R)
    net = nn.Sequential(nn.Linear(X.shape[1], 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))
    opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4)
    tr = np.flatnonzero(~va)
    for ep in range(40):
        for b in np.array_split(rng.permutation(tr), max(1, len(tr) // 256)):
            loss = (net(Xt[b])[:, 0] - Rt[b]).pow(2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        p = net(Xt)[:, 0].numpy()
    base = np.mean((R[va] - R[~va].mean()) ** 2); mse = np.mean((R[va] - p[va]) ** 2)
    print(f"{len(R)} launches ({RA.mean():.0%} random attitude), val {va.sum()}: MSE {mse:.2f} vs mean-only {base:.2f} (R^2 {1 - mse / base:.2f})")
    # attitude: best predicted vs the policy's own (non-random) choice, on held-out launches
    pol = va & (RA == 0)
    if pol.any():
        inc = np.linspace(-3.0, 3.0, 13); offs = np.linspace(-2.0, 2.0, 17)  # u grid (inclination 57~88 deg, offset +-173)
        best = np.full(pol.sum(), -np.inf); bu = np.zeros((pol.sum(), 2))
        with torch.no_grad():
            for a in inc:
                for b in offs:
                    q = net(torch.tensor((feats(O[pol], np.tile([a, b], (pol.sum(), 1))) - mu) / sd))[:, 0].numpy()
                    m = q > best; best[m] = q[m]; bu[m] = [a, b]
        print(f"policy attitude: predicted {p[pol].mean():.2f} pops; best attitude on the grid: {best.mean():.2f} "
              f"(+{(best - p[pol]).mean():.2f}); best inclination {np.mean(55 + 35 / (1 + np.exp(-bu[:, 0]))):.1f} deg vs "
              f"policy {np.mean(55 + 35 / (1 + np.exp(-U[pol, 0]))):.1f} deg")
        t = O[pol, 63] * 60.0
        for lo, hi in ((0, 10), (10, 20), (20, 30), (30, 61)):
            m = (t >= lo) & (t < hi)
            if m.any():
                print(f"  launch t {lo:>2}-{hi:<2}s: n {m.sum():4d}  actual {R[pol][m].mean():.2f}  predicted {p[pol][m].mean():.2f}")
    if out:
        torch.save({"net": net.state_dict(), "mu": mu, "sd": sd}, out)


if __name__ == "__main__":
    main(*sys.argv[1:])

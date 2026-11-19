"""Train the RL2 learned reach model from rl2_env decision logs (T1DATA/*.npy).
Row = rocket globals (6) + token (15) + stretch index | hit, time from decision to hit/abandon.

.venv/bin/python dev/train_t1sl.py 'runs/ppo5/t1data/*.npy' runs/ppo5/rl2_sl.npz
"""
import glob, os, sys
import numpy as np
import torch
sys.path.insert(0, ".")
from BalloonPoppingGymEnv.agents.rl2_agent import STRETCH  # noqa: E402
from BalloonPoppingGymEnv.agents.rl2_sl import IN_TOK  # noqa: E402

pattern, out = sys.argv[1], sys.argv[2]
MAX_ROWS = int(os.environ.get("SL_MAX_ROWS", 300_000))  # newest logs only: bounded training time, labels from the current policy
parts, n = [], 0
def newest(f):  # the write time: the time_ns in the name (<pid>_<time_ns>.npy, no NFS stat; 11/6 §483), else the mtime
    try:
        return int(os.path.basename(f).split("_")[1].split(".")[0]) / 1e9
    except (IndexError, ValueError):
        return os.path.getmtime(f)


for f in sorted({f for p in pattern.split("|") for f in glob.glob(p)}, key=newest, reverse=True):  # a|b: union (11/6)
    try:
        parts.append(np.load(f)); n += len(parts[-1])
    except ValueError:  # (a log still being written by a live worker)
        continue
    if n >= MAX_ROWS:
        break
R = np.concatenate(parts)
Xin = np.concatenate([R[:, :6], R[:, 6:21][:, IN_TOK]], 1).astype(np.float32)
s = R[:, 21].astype(int); hit = R[:, 22]; t = R[:, 23]
INIT = os.environ.get("SL_INIT")  # warm start: continue from a previous model (and keep ITS input normalisation,
EPOCHS, LR = int(os.environ.get("SL_EPOCHS", 30)), float(os.environ.get("SL_LR", 1e-3))  # so outputs move little)
if INIT:
    D0 = dict(np.load(INIT)); mu, sd = D0["mu"], D0["sd"]
else:
    mu, sd = Xin.mean(0), Xin.std(0) + 1e-6
X = torch.tensor((Xin - mu) / sd); S = torch.tensor(s); H = torch.tensor(hit, dtype=torch.float32); T = torch.tensor(t, dtype=torch.float32)
ns = len(STRETCH)
HID = [int(h) for h in os.environ.get("SL_HID", "256,256,256").split(",")]  # hidden widths (2026-09-25 18:50: 128x2 -> 256x3, val acc +0.8 pt, t_hit MAE -20%)
mods, d_in = [], X.shape[1]
for h in HID:
    mods += [torch.nn.Linear(d_in, h), torch.nn.ReLU()]; d_in = h
net = torch.nn.Sequential(*mods, torch.nn.Linear(d_in, 2 * ns))
if INIT and (int(D0["n_layers"]) != len(HID) + 1 or D0["W0"].shape[1] != HID[0]):
    if os.environ.get("SL_ALLOW_SCRATCH") != "1":  # 2026-09-26: this fallback silently turned "warm starts" into
        sys.exit(f"SL_INIT has a different shape ({int(D0['n_layers']) - 1} x {D0['W0'].shape[1]} vs SL_HID {HID}): "
                 "refusing to train from scratch (set SL_HID to match, or SL_ALLOW_SCRATCH=1)")  # from-scratch swaps
    print("SL_INIT has a different shape: training from scratch (its input normalisation is kept)"); INIT = None
    EPOCHS, LR = 30, 1e-3  # a full from-scratch schedule, not the short warm-start one
if INIT:
    with torch.no_grad():
        for i, m in enumerate([m for m in net if isinstance(m, torch.nn.Linear)]):
            m.weight.copy_(torch.tensor(D0[f"W{i}"].T)); m.bias.copy_(torch.tensor(D0[f"b{i}"]))
opt = torch.optim.Adam(net.parameters(), LR)
n = len(X); perm = torch.randperm(n); va, tr = perm[: n // 10], perm[n // 10 :]
if INIT:  # tracking check: hold out the NEWEST 10% of rows (logs are loaded newest first) -> predicts forward in time
    va, tr = torch.arange(n // 10), torch.arange(n // 10, n)
for ep in range(EPOCHS):
    for j in tr[torch.randperm(len(tr))].split(1024):
        o = net(X[j]); r = torch.arange(len(j))
        lp = torch.nn.functional.binary_cross_entropy_with_logits(o[r, S[j]], H[j])
        m = H[j] > 0
        lt = torch.nn.functional.smooth_l1_loss(o[r, ns + S[j]][m], T[j][m]) if m.any() else 0.0
        loss = lp + 0.1 * lt
        opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
    o = net(X[va]); r = torch.arange(len(va))
    p = torch.sigmoid(o[r, S[va]]); m = H[va] > 0
    print(f"rows {n}  hit rate {hit.mean():.3f}  val acc {((p > 0.5).float() == H[va]).float().mean():.3f}  "
          f"t_hit MAE on hits {(o[r, ns + S[va]][m] - T[va][m]).abs().mean():.2f}s")
lin = [m for m in net if isinstance(m, torch.nn.Linear)]
if INIT:  # how far the outputs the policy sees move (best stretch level, as rl2_sl.load_sl) + forward-in-time AUC
    L0 = [(D0[f"W{i}"], D0[f"b{i}"]) for i in range(int(D0["n_layers"]))]
    L1 = [(m.weight.detach().numpy().T, m.bias.detach().numpy()) for m in lin]
    xv, sv, yv = X[va].numpy(), s[va.numpy()], hit[va.numpy()]
    def fwd(Ws):
        z = xv
        for W, b in Ws[:-1]:
            z = np.maximum(z @ W + b, 0)
        o = z @ Ws[-1][0] + Ws[-1][1]; return 1 / (1 + np.exp(-o[:, :ns])), np.maximum(o[:, ns:], 0.0)
    def best(p, t):
        k = np.argmax(p - 0.02 * t, 1); r = np.arange(len(p)); return p[r, k], t[r, k]
    def auc(p, y):
        o = np.argsort(p); rk = np.empty(len(p)); rk[o] = np.arange(1, len(p) + 1); n1 = y.sum(); n0 = len(y) - n1
        return (rk[y > 0].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1)
    (P0, T0), (P1, T1) = fwd(L0), fwd(L1); r = np.arange(len(xv))
    (p0, t0), (p1, t1) = best(P0, T0), best(P1, T1); dp = np.abs(p1 - p0).mean()
    print(f"shift vs init: mean |dp| {dp:.4f}  mean |dt| {np.abs(t1 - t0).mean():.3f}s  mean p {p0.mean():.3f} -> {p1.mean():.3f}"
          f"  | newest rows: hit rate {yv.mean():.3f}  AUC {auc(P0[r, sv], yv):.3f} -> {auc(P1[r, sv], yv):.3f}")
    if not dp <= float(os.environ.get("SL_MAX_DP", "inf")):  # swap guard (NaN fails too): too big a change for the policy
        sys.exit(f"shift {dp:.4f} > SL_MAX_DP {os.environ['SL_MAX_DP']}: not saved, the deployed model stays")
out_d = {"n_layers": len(lin), "mu": mu, "sd": sd, "n_str": ns}
for i, m in enumerate(lin):
    out_d[f"W{i}"], out_d[f"b{i}"] = m.weight.detach().numpy().T, m.bias.detach().numpy()
np.savez(out, **out_d); print("saved", out)

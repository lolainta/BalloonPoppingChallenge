"""V2: body-rate distribution in burn, gimbal usage (ours logged; theirs inferred via own K fit), throttle proxy."""
import numpy as np
from base import *

def burn_sel(d, edge=0.5):
    b = d["burnmask"].copy(); b[:d["i0"] + int(edge / DT)] = False; b[-3:] = False
    return b

# --- own K fit: dw/dt = K * a_z * sin(g_actual), g_actual = rate-limited (0.6 deg/step) command, try lags
def slew(cmd, lim=0.6):
    y = np.zeros_like(cmd); prev = np.zeros(cmd.shape[1])
    for k in range(len(cmd)):
        prev = prev + np.clip(cmd[k] - prev, -lim, lim); y[k] = prev
    return y

res = {}
for lag in (-1, 0, 1, 2):
    xs, ys = [], []
    for n in NAMES:
        d = get("ours_strict", n)
        b = burn_sel(d, 0.1)
        az = (d["f"] * d["zb"]).sum(1)
        A = np.nan_to_num(d["A"][:, :2]); ga = slew(A)
        dw = np.full((len(d["t"]), 2), np.nan); dw[1:] = np.diff(d["w"][:, :2], axis=0) / DT  # dw[k] = w[k]-w[k-1]
        gs = np.roll(ga, lag, axis=0)  # use action from step k-lag
        # RocketPy: M1 = thrust2 * T * lever = sin(gx) T l  -> w1 ; M2 = -thrust1 T l = sin(gy) T l -> w2
        for ax in (0, 1):
            xs.append((az * np.sin(np.radians(gs[:, ax])))[b]); ys.append(dw[b, ax])
    x, y = np.concatenate(xs), np.concatenate(ys)
    ok = np.isfinite(x) & np.isfinite(y)
    K = (x[ok] @ y[ok]) / (x[ok] @ x[ok]); r2 = 1 - ((y[ok] - K * x[ok]) ** 2).mean() / y[ok].var()
    res[lag] = (K, r2)
    print(f"lag {lag}: K {K:.3f} 1/m  R2 {r2:.3f}")
lag = max(res, key=lambda k: res[k][1]); K = res[lag][0]
print("best lag", lag, "K", round(K, 3))

for kind in ("their", "ours_strict", "ours_free"):
    W, GI, GL, AZ = [], [], [], []
    per = []
    for n in NAMES:
        d = get(kind, n); b = burn_sel(d)
        W.append(d["wxy"][b]); per.append(np.percentile(d["wxy"][b], 95))
        az = (d["f"] * d["zb"]).sum(1)
        dw = np.full((len(d["t"]), 2), np.nan); dw[1:] = np.diff(d["w"][:, :2], axis=0) / DT
        dw = np.roll(dw, -lag, axis=0)  # dw caused by the gimbal at k
        gi = np.degrees(np.arcsin(np.clip(dw / (K * np.maximum(az, 1e-3))[:, None], -1, 1)))
        GI.append(gi[b]); AZ.append(az[b])
        if kind != "their":
            GL.append(np.nan_to_num(d["A"][b, :2]))
    W = np.concatenate(W); GI = np.concatenate(GI)
    gn = np.hypot(GI[:, 0], GI[:, 1])
    s = f"[{kind}] |w_xy| p50 {np.percentile(W,50):.2f} p90 {np.percentile(W,90):.2f} p95 {np.percentile(W,95):.2f} max {W.max():.2f}  >1.5: {np.mean(W>1.5):.3f}  >2: {np.mean(W>2):.3f}"
    s += f"\n   per-run p95 min {min(per):.2f} max {max(per):.2f}"
    s += f"\n   inferred gimbal |g| >=14: {np.nanmean(gn>=14):.3f}  >=14.5: {np.nanmean(gn>=14.5):.3f}  >16: {np.nanmean(gn>16):.3f}   p50 {np.nanmedian(gn):.2f}"
    if GL:
        GL = np.concatenate(GL); gl = np.hypot(GL[:, 0], GL[:, 1])
        s += f"\n   logged cmd |g| >=14.5: {np.mean(gl>=14.5):.3f}  >16: {np.mean(gl>16):.3f}  p50 {np.median(gl):.2f}"
    print(s)

"""V4: ours only: nose vs wanted F; phases; re-point delay after target changes; gimbal slew-limit fraction."""
import numpy as np
from base import *

for kind in ("ours_strict", "ours_free"):
    TH, FM, LOST, EV, PH, SLEW, WS = [], [], [], [], [], [], []
    for n in NAMES:
        d = get(kind, n); t = d["t"]; F = d["F"]; tgt = d["TGT"]
        g = d["burnmask"] & np.isfinite(F).all(1); g[:d["i0"] + 50] = False; g[-3:] = False
        th = np.full(len(t), np.nan); th[g] = angd(d["zb"][g], F[g])
        fm = np.linalg.norm(np.nan_to_num(F), axis=1)
        TH.append(th[g]); FM.append(fm[g]); LOST.append(np.sum(fm[g] * (1 - np.cos(np.radians(th[g])))) * DT)
        # phases: a = within 1 s after a target switch, b = last 0.5 s before a pop, c = rest
        sw = np.flatnonzero((tgt[1:] != tgt[:-1]) & (tgt[1:] >= 0)) + 1
        since = np.full(len(t), np.inf)
        for s in sw:
            e = min(s + 101, len(t)); since[s:e] = np.minimum(since[s:e], t[s:e] - t[s])
        topop = np.full(len(t), np.inf)
        for tp, _ in d["pops"]:
            k = ti(t, tp); ks = max(0, k - 50); topop[ks:k + 1] = np.minimum(topop[ks:k + 1], tp - t[ks:k + 1])
        ph = np.where(since <= 1.0, 0, np.where(topop <= 0.5, 1, 2))
        PH.append(np.stack([ph[g], th[g], fm[g]], 1))
        # re-point events: target change (to a valid target) during guided burn
        for e in sw:
            if not g[e]: continue
            jj = np.arange(e, min(e + 400, len(t)))
            stop = np.flatnonzero((tgt[jj] != tgt[e]) | ~g[jj])
            jj = jj[:stop[0]] if len(stop) else jj
            if len(jj) < 3: continue
            x = th[jj]; tt = t[jj] - t[e]
            EV.append((x[0], tt[np.argmax(x < 20)] if (x < 20).any() else np.nan, tt[np.argmax(x < 10)] if (x < 10).any() else np.nan))
        # gimbal slew-limited steps: |cmd[k] - cmd_applied[k-1]| > 0.6 deg (reconstructed actual)
        A = np.nan_to_num(d["A"][:, :2]); y = np.zeros(2); lim = np.zeros(len(t), bool)
        for k in range(len(t)):
            dlt = A[k] - y; lim[k] = (np.abs(dlt) > 0.6 + 1e-9).any(); y = y + np.clip(dlt, -0.6, 0.6)
        bb = d["burnmask"].copy(); bb[:d["i0"] + 50] = False
        SLEW.append(lim[bb])
    TH, FM = np.concatenate(TH), np.concatenate(FM)
    print(f"[{kind}] angle(nose,F) p50 {np.median(TH):.2f} mean {TH.mean():.1f} frac>20 {np.mean(TH>20):.3f} frac>10 {np.mean(TH>10):.3f}; eff {np.sum(FM*np.cos(np.radians(TH)))/FM.sum():.3f}; lost/run {np.mean(LOST):.1f} m/s")
    PH = np.concatenate(PH); m20 = PH[:, 1] > 20
    lost = PH[:, 2] * (1 - np.cos(np.radians(PH[:, 1])))
    for p, lab in enumerate(("<=1s after switch", "last 0.5s before pop", "rest")):
        s = PH[:, 0] == p
        print(f"   {lab:<22} time {s.mean():.3f}  share>20 {(s & m20).sum()/m20.sum():.3f}  share lost {lost[s].sum()/lost.sum():.3f}")
    EV = np.array(EV); big = EV[:, 0] > 30
    print(f"   events n={len(EV)} big={big.sum()}  t20 med {np.nanmedian(EV[big,1]):.2f}  t10 med {np.nanmedian(EV[big,2]):.2f} p90 {np.nanpercentile(EV[big,2],90):.2f}  (never<10: {np.isnan(EV[big,2]).sum()})")
    SLEW = np.concatenate(SLEW)
    print(f"   gimbal slew-limited fraction of burn steps: {SLEW.mean():.3f}")

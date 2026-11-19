"""Independent recompute of the decomp hop-start numbers (verifier)."""
import glob, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path('./'); D1 = ROOT/'data/gnc2'; D4 = ROOT/'dev/gnc2_seq/decomp/data'
g = np.array([0, 0, -9.81]); RH = 1.5
TG = np.arange(0.002, 15.0, 0.002)
def tmin(p, v, bp, bv, a, r=RH):
    t = TG[:, None, None]
    m = (bp - p)[None] + (bv - v)[None]*t - 0.5*g*t**2
    ok = np.linalg.norm(m, axis=2) <= 0.5*a*TG[:, None]**2 + r
    return np.where(ok.any(0), TG[ok.argmax(0)], 15.0)
def acap(t, R, launch, burn=30.0, mode='p95'):
    live = ~np.isnan(R[:, 0]); i = np.flatnonzero(live)
    v = R[i, 3:6]; acc = np.gradient(v, t[i], axis=0)
    rho = 1.225*np.clip(1-2.2558e-5*R[i, 2:3], .5, 1)**4.2559
    f = acc - g + 2.5937e-4*rho*np.linalg.norm(v, axis=1, keepdims=True)*v
    fm = np.linalg.norm(f, axis=1); tt = t[i]
    sel = (tt > launch+.5) & (tt < launch+burn-.5)
    return float(np.percentile(fm[sel], 95)), float(np.percentile(fm[sel], 99)), float(np.median(fm[sel]))
def hops(path, burn=30.0, afix=None, shift=0):
    z = np.load(path); t, R, B, S = z['t'], z['R'], z['B'], z['S']; launch = float(z['launch'])
    a95, a99, amed = acap(t, R, launch, burn); a = afix if afix else a95
    first = {b: int(np.argmax(S[:, b] == 2)) for b in range(S.shape[1]) if (S[:, b] == 2).any()}
    P = sorted((t[j], b, j) for b, j in first.items())
    P = [x for x in P if x[0] <= launch + burn]
    # check against stored pops
    zp = sorted(map(tuple, z['pops'][:, :2].tolist()))
    rows = []
    for k in range(len(P)-1):
        t0, b0, j = P[k]; t1, b1, _ = P[k+1]
        j0 = j - 1 + shift
        p, v = R[j0, :3], R[j0, 3:6]
        st = S[j0].copy(); st[b0] = 2
        # also mark any balloon that popped at the same step as b0
        for bb, jj in first.items():
            if jj <= j: st[bb] = 2
        al = np.flatnonzero(st == 1)
        tm = tmin(p, v, B[j0, al, :3].astype(float), B[j0, al, 3:6].astype(float), a)
        tc = float(tmin(p, v, B[j0, [b1], :3].astype(float), B[j0, [b1], 3:6].astype(float), a)[0])
        o = np.sort(tm); pk = lambda i: o[i] if len(o) > i else 15.0
        rows.append(dict(T=t1-t0, T_ch=tc, T_b1=pk(0), T_b2=pk(1), T_b3=pk(2), T_b5=pk(4), exc=tc-pk(0),
                         rank=(1+(tm < tc-1e-9).sum()) if S[j0, b1] == 1 else np.nan, st1=int(S[j0, b1]),
                         n10=(tm < 1).sum(), n15=(tm < 1.5).sum(), n20=(tm < 2).sum(), alt=p[2]-20, spd=np.linalg.norm(v),
                         a95=a95, a99=a99, amed=amed, same_step=sum(1 for jj in first.values() if jj == j), tbl=launch+burn-t0,
                         nground=int((S[j0] == 0).sum())))
    return rows, a95

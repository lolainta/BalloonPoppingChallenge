"""Own hop table: consecutive pops from S transitions; T0 with own reach (thrust from THEIR |dv/dt-g| envelope, time
varying); excess = hop - T0. Writes hoptab.pkl."""
import sys, pickle
sys.path.insert(0, __file__.rsplit('/', 1)[0])
import numpy as np, pandas as pd
from vg import *

# thrust model: per 0.5 s bin p99 of their |dv/dt - g| (launch+0.2 s .. burnout), then fit 1/a linear in t
tl_all, f_all = [], []
for n in NAMES:
    d = load('their', n); t, R, L = d['t'], d['R'], d['L']
    a = np.linalg.norm(np.gradient(R[:, 3:6], t, axis=0) - G, axis=1)
    m = (t - L > 0.2) & (t - L < 29.8) & np.isfinite(a)
    tl_all += list(t[m] - L); f_all += list(a[m])
tl_all, f_all = np.array(tl_all), np.array(f_all)
bins = np.arange(0, 30.01, 0.5); c, e = [], []
for i in range(len(bins) - 1):
    s = (tl_all >= bins[i]) & (tl_all < bins[i + 1])
    if s.sum() > 50:
        c.append(0.5 * (bins[i] + bins[i + 1])); e.append(np.percentile(f_all[s], 99))
c, e = np.array(c), np.array(e)
k = np.polyfit(c, 1 / e, 1)
TT = np.arange(0, 30.01, 0.1); AT = np.interp(TT, c, e)  # direct envelope (fit only reported)
print('bins', np.round(c[::6],2), np.round(e[::6],2))
set_athrust(TT, AT)
print('thrust model a(0)=%.2f a(15)=%.2f a(29)=%.2f  (fit resid max %.2f)' % (AT[0], AT[150], AT[290], np.max(np.abs(1 / np.polyval(k, c) - e))))
pickle.dump((TT, AT), open(__file__.rsplit('/', 1)[0] + '/athrust.pkl', 'wb'))

rows = []
for n in NAMES:
    for kind in ('their', 'ours_strict', 'ours_free'):
        d = load(kind, n); t, R, B, L, P = d['t'], d['R'], d['B'], d['L'], d['P']
        for j in range(1, len(P)):
            ta, ba, ia = P[j - 1]; tb, bb, ib = P[j]
            tl = t[ia] - L
            T0, u = reach(R[ia, :3], R[ia, 3:6], B[ia, bb, :3], B[ia, bb, 3:], tl) if tl < BURN else (np.inf, None)
            z = rotz(R[ia, 6:10])
            vhat = R[ia, 3:6] / np.linalg.norm(R[ia, 3:6])
            rel = B[ia, bb, :3] - R[ia, :3]
            row = dict(kind=kind, name=n, seed=SEED[n], j=j, ba=int(ba), bb=int(bb), ta=round(float(ta), 2), tb=round(float(tb), 2),
                       hop=tb - ta, tl=tl, tlb=t[ib] - L, T0=T0, dist0=np.linalg.norm(rel), speed0=np.linalg.norm(R[ia, 3:6]),
                       ang_nose=np.degrees(np.arccos(np.clip(z @ u, -1, 1))) if u is not None else np.nan,
                       ang_v_next=np.degrees(np.arccos(np.clip(vhat @ rel / np.linalg.norm(rel), -1, 1))))
            if kind != 'their':
                TGT, ST = d['TGT'], d['STRETCH']
                row['tgt_at_pop'] = int(TGT[ia]); row['tgt_after'] = int(TGT[min(ia + 1, len(t) - 1)])
                seg = TGT[ia + 1:ib + 1]; m = seg >= 0
                row['s_mean'] = float(ST[ia + 1:ib + 1][m].mean()) if m.any() else np.nan
            rows.append(row)
h = pd.DataFrame(rows); h['excess'] = h.hop - h.T0
pickle.dump(h, open(__file__.rsplit('/', 1)[0] + '/hoptab.pkl', 'wb'))
pd.set_option('display.width', 250)
print('raw hops per kind:', h.groupby('kind').size().to_dict())
print('post-burn (tlb>30.5):', h[h.tlb > 30.5].groupby('kind').size().to_dict(), ' infinite T0:', h[~np.isfinite(h.T0)].groupby('kind').size().to_dict())
g = h[(h.tlb <= 30.5) & np.isfinite(h.T0)]
dd = g.drop_duplicates(['kind', 'seed', 'ba', 'bb', 'ta', 'tb'])
for lab, df in (('ALL', g), ('DEDUP', dd)):
    print('==', lab)
    for kind in ('their', 'ours_strict', 'ours_free'):
        x = df[df.kind == kind]
        print(f"{kind:12s} n {len(x):3d} hop mean {x.hop.mean():.2f} med {x.hop.median():.2f} | T0 mean {x.T0.mean():.2f} med {x.T0.median():.2f} | excess mean {x.excess.mean():.2f} med {x.excess.median():.2f} | nose-err med {x.ang_nose.median():.1f} | v->next med {x.ang_v_next.median():.1f}")

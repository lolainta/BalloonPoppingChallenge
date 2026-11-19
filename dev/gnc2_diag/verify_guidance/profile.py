"""Loss profile L(tau) = tau + T1(ta+tau) - T1(ta) per dedup hop (own reach), nose / specific-force pointing vs the ideal
min-time direction, time to align within 10 deg. Also a frame sanity check: angle(body z, dv/dt - g) during burn."""
import sys, pickle
sys.path.insert(0, __file__.rsplit('/', 1)[0])
import numpy as np, pandas as pd
from vg import *
TT, AT = pickle.load(open(__file__.rsplit('/', 1)[0] + '/athrust.pkl', 'rb')); set_athrust(TT, AT)
h = pickle.load(open(__file__.rsplit('/', 1)[0] + '/hoptab.pkl', 'rb'))
h = h[(h.tlb <= 30.5) & np.isfinite(h.T0)].drop_duplicates(['kind', 'seed', 'ba', 'bb', 'ta', 'tb'])
rows, prof, frame = [], [], []
for (kind, n), g in h.groupby(['kind', 'name']):
    d = load(kind, n); t, R, B, L = d['t'], d['R'], d['B'], d['L']
    sf = np.gradient(R[:, 3:6], t, axis=0) - G
    sfs = np.stack([np.convolve(sf[:, c], np.ones(5) / 5, 'same') for c in range(3)], 1)
    z = rotz(R[:, 6:10])
    burn = (t - L > 0.3) & (t - L < 29.7) & np.isfinite(R[:, 0])
    cz = np.einsum('ij,ij->i', z[burn], sfs[burn]) / np.linalg.norm(sfs[burn], axis=1)
    frame.append((kind, np.degrees(np.arccos(np.clip(cz, -1, 1)))))
    for _, r in g.iterrows():
        ia = int(np.searchsorted(t, r.ta - 1e-6)); ib = int(np.searchsorted(t, r.tb - 1e-6))
        T0 = None; Ls = {}; al_body = al_sf = None
        for i in range(ia, ib + 1, 5):
            tl = t[i] - L
            T, u = reach(R[i, :3], R[i, 3:6], B[i, r.bb, :3], B[i, r.bb, 3:], tl)
            if T0 is None: T0 = T
            tau = t[i] - t[ia]
            Ls[round(tau, 2)] = tau + T - T0
            if u is not None:
                ab = np.degrees(np.arccos(np.clip(z[i] @ u, -1, 1)))
                af = np.degrees(np.arccos(np.clip(sfs[i] @ u / np.linalg.norm(sfs[i]), -1, 1)))
                if al_body is None and ab < 10: al_body = tau
                if al_sf is None and af < 10: al_sf = tau
                prof.append(dict(kind=kind, tau=tau, rem=t[ib] - t[i], ab=ab, af=af))
        taus = np.array(sorted(Ls)); Lv = np.array([Ls[k] for k in taus])
        hop = t[ib] - t[ia]
        rows.append(dict(kind=kind, name=n, bb=r.bb, hop=hop, T0=T0, excess=hop - T0, s_mean=r.get('s_mean', np.nan),
                         L05=np.interp(min(0.5, hop), taus, Lv), L10=np.interp(min(1.0, hop), taus, Lv),
                         Lm05=np.interp(max(hop - 0.5, 0), taus, Lv), align_body=al_body if al_body is not None else hop,
                         align_sf=al_sf if al_sf is not None else hop, ang0=r.ang_nose))
p = pd.DataFrame(rows)
pickle.dump((p, pd.DataFrame(prof)), open(__file__.rsplit('/', 1)[0] + '/profile.pkl', 'wb'))
for kind, a in frame:
    pass
fr = {}
for kind, a in frame: fr.setdefault(kind, []).append(a)
for kind in fr:
    a = np.concatenate(fr[kind]); print(f'frame check {kind}: angle(body z, smoothed dv/dt-g) p50 {np.percentile(a,50):.1f} p90 {np.percentile(a,90):.1f} p99 {np.percentile(a,99):.1f}')
for kind in ('their', 'ours_strict', 'ours_free'):
    x = p[p.kind == kind]; y = x[x.hop < 8]
    print(f"{kind:12s} n {len(x)} L(0.5) {x.L05.mean():.2f} L(1.0) {x.L10.mean():.2f} L(tb-0.5) {x.Lm05.mean():.2f} excess {x.excess.mean():.2f} | hop<8: n {len(y)} {y.L05.mean():.2f} {y.L10.mean():.2f} {y.Lm05.mean():.2f} {y.excess.mean():.2f} | ang0 med {x.ang0.median():.1f} align10 body med {x.align_body.median():.2f} sf med {x.align_sf.median():.2f}")

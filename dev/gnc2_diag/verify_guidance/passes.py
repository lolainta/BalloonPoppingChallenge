"""Distance local minima to b_k inside each dedup hop (own code): near-miss re-passes (1.5 < d < 6 m) and flybys (6-40 m);
time lost = tb - t(first near-miss). Also ours: was b_k the GNC target at that minimum?"""
import sys, pickle
sys.path.insert(0, __file__.rsplit('/', 1)[0])
import numpy as np, pandas as pd
from vg import *
h = pickle.load(open(__file__.rsplit('/', 1)[0] + '/hoptab.pkl', 'rb'))
h = h[(h.tlb <= 30.5) & np.isfinite(h.T0)].drop_duplicates(['kind', 'seed', 'ba', 'bb', 'ta', 'tb'])
rows = []
for (kind, n), g in h.groupby(['kind', 'name']):
    d = load(kind, n); t, R, B = d['t'], d['R'], d['B']
    for _, r in g.iterrows():
        ia = int(np.searchsorted(t, r.ta - 1e-6)); ib = int(np.searchsorted(t, r.tb - 1e-6))
        dist = np.linalg.norm(B[ia:ib + 1, r.bb, :3] - R[ia:ib + 1, :3], axis=1)
        k = np.flatnonzero((dist[1:-1] < dist[:-2]) & (dist[1:-1] <= dist[2:])) + 1
        k = k[k < len(dist) - 2]
        near = [(t[ia + j] - t[ia], dist[j], (d['TGT'][ia + j] == r.bb) if kind != 'their' else None) for j in k if 1.5 < dist[j] < 6]
        fly = [(t[ia + j] - t[ia], dist[j]) for j in k if 6 <= dist[j] < 40]
        rows.append(dict(kind=kind, name=n, ba=r.ba, bb=r.bb, ta=r.ta, hop=r.hop, T0=r.T0, excess=r.excess, dist0=r.dist0,
                         near=near, fly=fly, lost=(r.hop - near[0][0]) if near else 0.0, s_mean=r.get('s_mean', np.nan)))
p = pd.DataFrame(rows)
for kind in ('their', 'ours_strict', 'ours_free'):
    x = p[p.kind == kind]
    nn = x[x.near.map(len) > 0]; ff = x[x.fly.map(len) > 0]
    print(f"{kind:12s} n {len(x)} near-miss hops {len(nn)} lost sum {x.lost.sum():.2f}s = {x.lost.sum() / len(x):.3f}s/hop | flyby hops {len(ff)}")
    for _, r in nn.iterrows(): print('   near', r['name'], r.ba, '->', r.bb, 'hop %.2f T0 %.2f' % (r.hop, r.T0), [(round(a, 2), round(b, 2), c) for a, b, c in r.near], 'lost %.2f s_mean %.2f' % (r.lost, r.s_mean))
    for _, r in ff.iterrows(): print('   fly ', r['name'], r.ba, '->', r.bb, 'hop %.2f T0 %.2f exc %.2f d0 %.1f' % (r.hop, r.T0, r.excess, r.dist0), [(round(a, 2), round(b, 1)) for a, b in r.fly])

"""Own analysis of the guidance_law sims: hop / T0 / excess / L(1.0) per variant (dedup by seed, ba, bb, ta, tb), matched
hops vs base strict, flights/seed-level pop deltas."""
import sys, pickle, glob
sys.path.insert(0, __file__.rsplit('/', 1)[0])
import numpy as np, pandas as pd
from vg import *
TT, AT = pickle.load(open(__file__.rsplit('/', 1)[0] + '/athrust.pkl', 'rb')); set_athrust(TT, AT)
SIM = ROOT / 'dev/gnc2_diag/guidance_law/sims'
INFO = {i['name']: i for i in __import__('json').load(open(D / 'replay_info.json'))}


def hops(path, n):
    f = np.load(path); t, R, B, S = f['t'], f['R'], f['B'].astype(float), f['S']; L = float(f['launch'])
    P = sorted((t[np.flatnonzero(S[:, b] == 2)[0]], b, np.flatnonzero(S[:, b] == 2)[0]) for b in range(S.shape[1]) if (S[:, b] == 2).any())
    out = []
    for j in range(1, len(P)):
        (ta, a, ia), (tb, b, ib) = P[j - 1], P[j]
        if t[ib] - L > 30.5: continue
        tl = t[ia] - L
        T0, u = reach(R[ia, :3], R[ia, 3:6], B[ia, b, :3], B[ia, b, 3:], tl)
        T1, _ = reach(R[min(ia + 100, ib), :3], R[min(ia + 100, ib), 3:6], B[min(ia + 100, ib), b, :3], B[min(ia + 100, ib), b, 3:], t[min(ia + 100, ib)] - L)
        tau1 = t[min(ia + 100, ib)] - t[ia]
        out.append(dict(name=n, seed=SEED[n], ba=int(a), bb=int(b), ta=round(float(ta), 2), tb=round(float(tb), 2), hop=tb - ta, T0=T0,
                        L10=tau1 + T1 - T0, nose=np.degrees(np.arccos(np.clip(rotz(R[ia, 6:10]) @ u, -1, 1))) if u is not None else np.nan))
    return out, len(P)


res = {}
for v in sys.argv[1:]:
    rows, npops = [], {}
    for n in NAMES:
        path = (D / f'ours_strict_{n}.npz') if v == 'base' else SIM / f'{v}__{n}.npz'
        if not path.exists(): continue
        r, k = hops(path, n); rows += r; npops[n] = k
    h = pd.DataFrame(rows); h = h[np.isfinite(h.T0)]
    dd = h.drop_duplicates(['seed', 'ba', 'bb', 'ta', 'tb']); dd = dd.assign(excess=dd.hop - dd.T0)
    res[v] = dd
    print(f"{v:6s} pops {sum(npops.values())} hops dedup {len(dd)} hop mean {dd.hop.mean():.2f} med {dd.hop.median():.2f} T0 {dd.T0.mean():.2f} excess {dd.excess.mean():.2f} L(1.0) {dd.L10.mean():.2f} nose0 med {dd.nose.median():.1f}")
if 'base' in res:
    b = res['base']
    for v in res:
        if v == 'base': continue
        m = b.merge(res[v], on=['seed', 'ba', 'bb'], suffixes=('_b', '_v'))
        m = m.drop_duplicates(['seed', 'ba', 'bb'])
        print(f"matched base vs {v}: n {len(m)} hop {m.hop_b.mean():.2f} -> {m.hop_v.mean():.2f} (d {(m.hop_v - m.hop_b).mean():+.2f}, med {(m.hop_v - m.hop_b).median():+.2f})  T0 {m.T0_b.mean():.2f} -> {m.T0_v.mean():.2f}  excess {m.excess_b.mean():.2f} -> {m.excess_v.mean():.2f}")

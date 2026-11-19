import sys; sys.path.insert(0, './dev/gnc2_seq/verify_decomp')
from v1 import *
rng = np.random.default_rng(5)
rows, npop, seeds = [], 0, []
for f in sorted(D4.glob('s4free_*.npz')):
    z = np.load(f); npop += len(z['pops']); seeds.append(int(z['seed']))
    rows += [dict(r, seed=int(z['seed'])) for r in hops(f, float(z['burn_time']))[0]]
d = pd.DataFrame(rows); print('files', len(seeds), 'unique seeds', len(set(seeds)), min(seeds), max(seeds), 'pops', npop, 'hops', len(d))
def sb(c, f=np.nanmean, B=4000):
    g = [x[c].to_numpy(float) for _, x in d.groupby('seed')]
    bs = [f(np.concatenate([g[i] for i in rng.integers(0, len(g), len(g))])) for _ in range(B)]
    return np.round([f(np.concatenate(g)), *np.percentile(bs, [2.5, 97.5])], 2)
for c in ['T_ch', 'T_b1', 'T_b2', 'T_b3', 'T_b5', 'exc', 'n10', 'n15', 'n20', 'rank', 'alt', 'spd']:
    print(c, 'mean', sb(c), 'median', sb(c, np.nanmedian))
r = d['rank']; print('rank1 %.2f top2 %.2f top3 %.2f mean %.2f' % ((r == 1).mean(), (r <= 2).mean(), (r <= 3).mean(), r.mean()))
print('a95 range', d.a95.min(), d.a95.max())

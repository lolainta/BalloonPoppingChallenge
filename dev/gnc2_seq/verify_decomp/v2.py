import sys; sys.path.insert(0, './dev/gnc2_seq/verify_decomp')
from v1 import *
names = '10_CoolBa 10_SpaceY 11_Mochi 11_near_miss 12_Apex 13_Apex 13_Yaka 14_SpaceBerry 16_SpaceBerry 16_TeamDJ 18_Taoyuan'.split()
keep = {'10_CoolBa', '10_SpaceY', '11_near_miss', '12_Apex', '14_SpaceBerry'}
def build(afix=None, shift=0):
    rows = []
    for n in names:
        for k in ('their', 'ours_free'):
            if k == 'ours_free' and n not in keep: continue
            z = np.load(D1/f'{k}_{n}.npz')
            rs, _ = hops(D1/f'{k}_{n}.npz', afix=afix, shift=shift)
            rows += [dict(r, kind=k, name=n, seed=int(z['seed'])) for r in rs]
    return pd.DataFrame(rows)
rng = np.random.default_rng(7)
def seedboot(d, col, f=np.nanmean, B=4000):
    g = [x[col].to_numpy(float) for _, x in d.groupby('seed')]
    bs = [f(np.concatenate([g[i] for i in rng.integers(0, len(g), len(g))])) for _ in range(B)]
    return f(np.concatenate(g)), *np.percentile(bs, [2.5, 97.5])
def paired(o, t, col, B=4000):
    d = (o.groupby('seed')[col].mean() - t.groupby('seed')[col].mean()).dropna().to_numpy()
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)]
    return d.mean(), *np.percentile(bs, [2.5, 97.5]), np.round(d, 2)
def report(df, tag):
    o, t = df[df.kind == 'ours_free'], df[df.kind == 'their']
    print(f'=== {tag}: hops ours {len(o)} theirs {len(t)}; same-step multi pops ours {(o.same_step>1).sum()} theirs {(t.same_step>1).sum()}')
    for c in ['T_ch', 'T_b1', 'T_b2', 'T_b3', 'T_b5', 'exc', 'rank', 'n10', 'n15', 'n20', 'alt', 'spd', 'T']:
        a = seedboot(o, c); b = seedboot(t, c); p = paired(o, t, c)
        print(f'{c:5s} ours {a[0]:.2f} [{a[1]:.2f},{a[2]:.2f}] med {o[c].median():.2f} | theirs {b[0]:.2f} [{b[1]:.2f},{b[2]:.2f}] med {t[c].median():.2f} | paired {p[0]:+.2f} [{p[1]:+.2f},{p[2]:+.2f}] per-seed {p[3]}')
    for nm, d in (('ours', o), ('theirs', t)):
        r = d['rank']; print(nm, 'rank1 %.2f top2 %.2f top3 %.2f mean %.2f flying %.2f' % ((r == 1).mean(), (r <= 2).mean(), (r <= 3).mean(), r.mean(), (d.st1 == 1).mean()))
if __name__ == '__main__':
    df = build(); df.to_csv('./dev/gnc2_seq/verify_decomp/v_hops_s1.csv', index=False); report(df, 'base (p95 a_cap, state j-1)')
    report(build(afix=14.2), 'common a_cap 14.2')
    report(build(shift=-10), 'state 0.1 s before pop')
    d3 = df[df.tbl >= 3]; report(d3, 'tb_left>=3')

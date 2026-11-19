import sys, numpy as np, pandas as pd
rng = np.random.default_rng(3)
def pb(o, t, c, B=4000):
    d = (o.groupby('seed')[c].mean() - t.groupby('seed')[c].mean()).dropna().to_numpy()
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)]; return d.mean(), *np.percentile(bs, [2.5, 97.5])
def sb(d, c, f=np.mean, B=4000):
    g = [x[c].to_numpy() for _, x in d.groupby('seed')]
    bs = [f(np.concatenate([g[i] for i in rng.integers(0, len(g), len(g))])) for _ in range(B)]; return f(np.concatenate(g)), *np.percentile(bs, [2.5, 97.5])
df = pd.read_csv(sys.argv[1])
for (K, e), d in df.groupby(['K', 'endv']):
    print(f'##### K={K} endv={e}')
    kinds = d.kind.unique()
    for k in kinds:
        x = d[d.kind == k]; s = f'{k}: n={len(x)} '
        for D in (1, 2, 3):
            m = sb(x, f'xs{D}'); s += f'| xs{D} mean {m[0]:.2f} [{m[1]:.2f},{m[2]:.2f}] /hop {m[0]/D:.2f} med {x[f"xs{D}"].median():.2f} P(>0.5) {(x[f"xs{D}"]>0.5).mean():.2f} ch {x[f"ch{D}"].mean():.2f} gy {x[f"gy{D}"].mean():.2f} gr med {x[f"gr{D}"].median():.2f} Copt/D {x[f"C{D}opt"].mean()/D:.2f} '
        print(s)
        # trimmed / capped
        for D in (2, 3):
            xs = x[f'xs{D}'].clip(upper=3.0); print(f'   xs{D} capped at 3 s mean {xs.mean():.2f}; C{D}opt median/D {x[f"C{D}opt"].median()/D:.2f}; C{D}opt>=15: {(x[f"C{D}opt"]>=15).sum()}')
    if 'their' in kinds:
        o, t = d[d.kind == 'ours_free'], d[d.kind == 'their']
        for c in ['xs1', 'xs2', 'xs3', 'C1opt', 'C2opt', 'C3opt', 'ch3']:
            print('  paired', c, np.round(pb(o, t, c), 2))
        o = o.assign(xs3c=o.xs3.clip(upper=3)); t = t.assign(xs3c=t.xs3.clip(upper=3))
        print('  paired xs3 capped3', np.round(pb(o, t, 'xs3c'), 2))

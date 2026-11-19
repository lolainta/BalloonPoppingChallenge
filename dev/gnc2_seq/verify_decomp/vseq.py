"""Independent depth-D chained lookahead regret (verifier). variants: endv='raw' (v+2m/t+Gt as in seq.py) or 'fix'
(miss reduced by the hit radius, force clipped to a), K first/next candidates."""
import sys, os
os.environ.setdefault('OMP_NUM_THREADS', '1')
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, './dev/gnc2_seq/verify_decomp')
from v1 import acap, D1, D4
g = np.array([0, 0, -9.81]); RH = 1.5; GR = np.arange(0.01, 15.0, 0.01); CAP = 15.0
def reach(p, v, bp, bv, a, endv):
    t = GR[:, None, None]
    m = (bp - p)[None] + (bv - v)[None]*t - 0.5*g*t**2
    nm = np.linalg.norm(m, axis=2)
    ok = nm <= 0.5*a*GR[:, None]**2 + RH
    i = ok.argmax(0); T = np.where(ok.any(0), GR[i], CAP)
    mm = m[i, np.arange(len(bp))]
    if endv == 'fix':
        n = np.linalg.norm(mm, axis=1, keepdims=True)
        mm = mm*np.clip((n - RH)/np.maximum(n, 1e-9), 0, 1)
        f = 2*mm/T[:, None]**2; fn = np.linalg.norm(f, axis=1, keepdims=True)
        f = f*np.minimum(1, a/np.maximum(fn, 1e-9))
        ve = v + (f + g)*T[:, None]
    else:
        ve = v + 2*mm/T[:, None] + g*T[:, None]
    return T, ve
def best(p, v, bp, bv, alive, a, D, K, endv):
    if D == 0 or len(alive) == 0: return 0.0
    T, ve = reach(p, v, bp[alive], bv[alive], a, endv); out = np.inf
    for j in np.argsort(T)[:K]:
        out = min(out, T[j] + best(bp[alive[j]] + bv[alive[j]]*T[j], ve[j], bp + bv*T[j], bv, np.delete(alive, j), a, D-1, K, endv))
    return out
def flight(path, K=8, endv='raw', Ds=(1, 2, 3), burn=30.0):
    z = np.load(path); t, R, B, S = z['t'], z['R'], z['B'], z['S']; launch = float(z['launch'])
    a = acap(t, R, launch, burn)[0]
    first = sorted((int(np.argmax(S[:, b] == 2)), b) for b in range(100) if (S[:, b] == 2).any())
    first = [x for x in first if t[x[0]] <= launch + burn]
    rows = []
    for k in range(len(first)-1):
        j, b0 = first[k]; b1 = first[k+1][1]; j0 = j - 1
        if S[j0, b1] != 1: continue
        st = S[j0].copy(); st[b0] = 2; alive = np.flatnonzero(st == 1)
        p, v = R[j0, :3], R[j0, 3:6]; bp, bv = B[j0, :, :3].astype(float), B[j0, :, 3:6].astype(float)
        ci = int(np.flatnonzero(alive == b1)[0])
        T1, ve = reach(p, v, bp[alive], bv[alive], a, endv); gi = int(np.argmin(T1))
        r = dict(tbl=launch+burn-t[j], T1ch=T1[ci], T1b=T1.min())
        for D in Ds:
            C = np.full(len(alive), np.inf)
            for i in set(np.argsort(T1)[:K].tolist()) | {ci, gi}:
                C[i] = T1[i] + best(bp[alive[i]] + bv[alive[i]]*T1[i], ve[i], bp + bv*T1[i], bv, np.delete(alive, i), a, D-1, K, endv)
            r.update({f'C{D}ch': C[ci], f'C{D}opt': C.min(), f'xs{D}': C[ci]-C.min(), f'gr{D}': C[gi]-C.min(),
                      f'ch{D}': float(C.argmin() == ci), f'gy{D}': float(C.argmin() == gi)})
        rows.append(r)
    return rows
names = '10_CoolBa 10_SpaceY 11_Mochi 11_near_miss 12_Apex 13_Apex 13_Yaka 14_SpaceBerry 16_SpaceBerry 16_TeamDJ 18_Taoyuan'.split()
keep = {'10_CoolBa', '10_SpaceY', '11_near_miss', '12_Apex', '14_SpaceBerry'}
def job(args):
    kind, n, path, K, endv = args
    return [dict(r, kind=kind, name=n, seed=int(np.load(path)['seed']), K=K, endv=endv) for r in flight(path, K, endv)]
if __name__ == '__main__':
    import multiprocessing as mp
    variants = [(8, 'raw'), (8, 'fix'), (12, 'raw')]
    which = sys.argv[1]
    jobs = []
    for K, e in variants:
        if which == 's1':
            jobs += [('their', n, D1/f'their_{n}.npz', K, e) for n in names] + [('ours_free', n, D1/f'ours_free_{n}.npz', K, e) for n in keep]
        else:
            jobs += [('s4', f.stem, f, K, e) for f in sorted(D4.glob('s4free_*.npz'))]
    with mp.get_context('spawn').Pool(8) as pool:
        res = pool.map(job, jobs)
    pd.DataFrame([r for rs in res for r in rs]).to_csv(f'./dev/gnc2_seq/verify_decomp/vseq_{which}.csv', index=False)

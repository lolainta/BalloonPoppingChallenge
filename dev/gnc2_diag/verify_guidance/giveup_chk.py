"""Terminal TGO jumps before each strict pop (own code, dedup flights): first step within 0.6 s before the pop where the
logged TGO rises by > 0.5 s vs the previous step while the popped balloon is the target; F (wanted specific force) rotation
over the last 0.5 s; body rate at pop; and whether the hop was a pop at all (it always is here)."""
import sys
sys.path.insert(0, __file__.rsplit('/', 1)[0])
import numpy as np
from vg import *
seen, n, nj, when, rot, rate, rot_nojump, jumps_hi = set(), 0, 0, [], [], [], [], []
for nm in NAMES:
    d = load('ours_strict', nm); t, R, TGO, TGT, F = d['t'], d['R'], d['TGO'], d['TGT'], d['F']
    for tp, b, i in d['P']:
        key = (SEED[nm], b, round(tp, 2))
        if key in seen: continue
        seen.add(key)
        if tp - d['L'] > 30: continue
        n += 1
        w = np.arange(max(i - 60, 1), i)
        on = TGT[w] == b
        dj = np.diff(TGO[max(i - 61, 0):i])
        jmp = [(t[i] - t[k]) for k, dd, o in zip(w, dj, on) if o and np.isfinite(dd) and dd > 0.5]
        if jmp: nj += 1; when.append(jmp[0])
        f0, f1 = F[i - 50], F[i - 1]
        if np.isfinite(f0).all() and np.isfinite(f1).all():
            a = np.degrees(np.arccos(np.clip(f0 @ f1 / np.linalg.norm(f0) / np.linalg.norm(f1), -1, 1)))
            rot.append(a); (rot_nojump if not jmp else jumps_hi).append(a)
        rate.append(np.linalg.norm(R[i, 10:13]))
print(f'strict pops (dedup, in burn) {n}; with a TGO jump >0.5 s in the last 0.6 s while targeted: {nj} ({100 * nj / n:.0f}%), first jump med {np.median(when):.2f} s before pop')
print(f'F rotation over last 0.5 s: med {np.median(rot):.0f} deg (with jump {np.median(jumps_hi):.0f}, without {np.median(rot_nojump) if rot_nojump else float("nan"):.0f}); body rate at pop med {np.median(rate):.2f} rad/s')
# theirs: body rate at pop for comparison
rr = []
seen = set()
for nm in NAMES:
    d = load('their', nm)
    for tp, b, i in d['P']:
        key = (SEED[nm], b, round(tp, 2))
        if key in seen or tp - d['L'] > 30: continue
        seen.add(key); rr.append(np.linalg.norm(d['R'][i, 10:13]))
print(f'their body rate at pop med {np.median(rr):.2f} rad/s (n {len(rr)})')

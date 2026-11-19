"""Matched balloon pairs b_{k-1}->b_k popped consecutively by both (strict vs theirs), dedup by (seed, ba, bb, our ta).
Velocity swap: T0 from our position with THEIR velocity at their pop of b_{k-1} (balloons at our time); also their position
with our velocity (reverse swap) and relative-time-shift check."""
import sys, pickle
sys.path.insert(0, __file__.rsplit('/', 1)[0])
import numpy as np, pandas as pd
from vg import *
TT, AT = pickle.load(open(__file__.rsplit('/', 1)[0] + '/athrust.pkl', 'rb')); set_athrust(TT, AT)
rows, seen = [], set()
for n in NAMES:
    th, us = load('their', n), load('ours_strict', n)
    tp = {b: (t, i) for t, b, i in th['P']}
    order = [b for _, b, _ in th['P']]
    for j in range(1, len(us['P'])):
        (ta, a, ia), (tb, b, ib) = us['P'][j - 1], us['P'][j]
        if a not in tp or b not in tp or order.index(b) != order.index(a) + 1:
            continue
        if us['t'][ib] - us['L'] > 30.5:
            continue
        key = (SEED[n], a, b, round(ta, 2))
        if key in seen: continue
        seen.add(key)
        (tta, ja), (ttb, jb) = tp[a], tp[b]
        Ru, Bu, Rt, Bt = us['R'], us['B'], th['R'], th['B']
        tlu, tlt = us['t'][ia] - us['L'], th['t'][ja] - th['L']
        T_own, _ = reach(Ru[ia, :3], Ru[ia, 3:6], Bu[ia, b, :3], Bu[ia, b, 3:], tlu)
        T_vth, _ = reach(Ru[ia, :3], Rt[ja, 3:6], Bu[ia, b, :3], Bu[ia, b, 3:], tlu)
        T_th, _ = reach(Rt[ja, :3], Rt[ja, 3:6], Bt[ja, b, :3], Bt[ja, b, 3:], tlt)
        T_th_vus, _ = reach(Rt[ja, :3], Ru[ia, 3:6], Bt[ja, b, :3], Bt[ja, b, 3:], tlt)
        rows.append(dict(name=n, a=a, b=b, hop_u=tb - ta, hop_t=ttb - tta, T_u=T_own, T_u_vth=T_vth, T_t=T_th, T_t_vus=T_th_vus,
                         late=ta - tta, dpos=np.linalg.norm(Ru[ia, :3] - Rt[ja, :3]), su=np.linalg.norm(Ru[ia, 3:6]), st=np.linalg.norm(Rt[ja, 3:6]),
                         dang=np.degrees(np.arccos(np.clip(Ru[ia, 3:6] @ Rt[ja, 3:6] / np.linalg.norm(Ru[ia, 3:6]) / np.linalg.norm(Rt[ja, 3:6]), -1, 1)))))
r = pd.DataFrame(rows); r = r[np.isfinite(r[['T_u', 'T_u_vth', 'T_t', 'T_t_vus']]).all(1)]
r['ex_u'] = r.hop_u - r.T_u; r['ex_t'] = r.hop_t - r.T_t
pd.set_option('display.width', 250)
print('n pairs', len(r), ' frac ours slower', (r.hop_u > r.hop_t).mean().round(2), ' ratio med', (r.hop_u / r.hop_t).median().round(2))
print(r[['hop_u', 'hop_t', 'T_u', 'T_u_vth', 'T_t', 'T_t_vus', 'ex_u', 'ex_t', 'late', 'dpos', 'su', 'st', 'dang']].agg(['mean', 'median']).round(2).to_string())
print('hop gap %.2f = T0 gap %.2f + excess gap %.2f' % ((r.hop_u - r.hop_t).mean(), (r.T_u - r.T_t).mean(), (r.ex_u - r.ex_t).mean()))
print('T0 gap split: velocity (T_u - T_u_vth) %.2f, rest (T_u_vth - T_t) %.2f; reverse swap: (T_t_vus - T_t) %.2f' % (
    (r.T_u - r.T_u_vth).mean(), (r.T_u_vth - r.T_t).mean(), (r.T_t_vus - r.T_t).mean()))
pickle.dump(r, open(__file__.rsplit('/', 1)[0] + '/pairs.pkl', 'wb'))

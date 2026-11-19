"""t_go estimate quality (own code): for each dedup strict hop, at the hop start (first step after the previous pop that
targets b_k) and at mid-hop: planned = STRETCH * TGO (what the guidance flies), actual = t_pop - t; grouped by STRETCH at
that step. Also actual / TGO. And TGO decrease rate per second in the middle of s=1.0 stretches."""
import sys
sys.path.insert(0, __file__.rsplit('/', 1)[0])
import numpy as np, pandas as pd
from vg import *
rows, seen = [], set()
for nm in NAMES:
    d = load('ours_strict', nm); t, TGO, TGT, ST = d['t'], d['TGO'], d['TGT'], d['STRETCH']
    P = d['P']
    for j in range(1, len(P)):
        (ta, a, ia), (tb, b, ib) = P[j - 1], P[j]
        key = (SEED[nm], a, b, round(ta, 2))
        if key in seen or tb - d['L'] > 30.5: continue
        seen.add(key)
        on = np.flatnonzero(TGT[ia + 1:ib] == b) + ia + 1
        if not len(on): continue
        for ph, i in (('start', on[0]), ('mid', on[len(on) // 2])):
            if np.isfinite(TGO[i]) and TGO[i] < 14.9:
                rows.append(dict(name=nm, ph=ph, s=ST[i], tgo=TGO[i], plan=ST[i] * TGO[i], act=tb - t[i]))
        # TGO slope where stretch==1 in the middle 50% of the hop
        mm = on[(t[on] > ta + 0.25 * (tb - ta)) & (t[on] < tb - 0.25 * (tb - ta)) & np.isclose(ST[on], 1.0)]
        if len(mm) > 20:
            k = np.polyfit(t[mm], TGO[mm], 1)[0]
            rows.append(dict(name=nm, ph='slope1', s=1.0, tgo=k, plan=np.nan, act=np.nan))
r = pd.DataFrame(rows)
r['act_plan'] = r.act / r.plan; r['act_tgo'] = r.act / r.tgo
pd.set_option('display.width', 200)
print(r[r.ph != 'slope1'].groupby(['ph', 's']).agg(n=('act', 'size'), act_plan_med=('act_plan', 'median'), act_plan_mean=('act_plan', 'mean'),
                                                   act_tgo_med=('act_tgo', 'median'), act_tgo_mean=('act_tgo', 'mean')).round(2).to_string())
sl = r[r.ph == 'slope1']
print('dTGO/dt where s=1 mid-hop: n %d mean %.2f med %.2f' % (len(sl), sl.tgo.mean(), sl.tgo.median()))

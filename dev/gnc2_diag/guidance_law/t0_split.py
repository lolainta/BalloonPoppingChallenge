"""Strict mode, same balloon pair (b_{k-1} -> b_k popped consecutively by both): how much of the T0 gap (ideal min time at
the previous pop, ours vs theirs) is the arrival VELOCITY? T1 from (our position, our velocity) vs (our position, THEIR
velocity at their pop of b_{k-1}), balloons at our time. Also: does the previous hop's speed level set the next T0?"""
import pickle
import numpy as np, pandas as pd
from chases import D, NAMES, reach, a_at, BURN
from hops import SEED

h = pd.DataFrame(pickle.load(open("hops.pkl", "rb")))
rows, seen = [], set()
for n in NAMES:
    th = np.load(D / f"their_{n}.npz"); us = np.load(D / f"ours_strict_{n}.npz")
    tt, Rt = th["t"], th["R"]; tu, Ru, Bu = us["t"], us["R"], us["B"].astype(float); L = float(us["launch"])
    ptheir = {int(b): tp for tp, b in th["pops"]}
    po = us["pops"]
    for k in range(1, len(po)):
        a, b = int(po[k - 1][1]), int(po[k][1])
        if a not in ptheir or b not in ptheir:
            continue
        # consecutive for them too?
        ordr = [int(x) for x in th["pops"][:, 1]]
        if ordr.index(b) != ordr.index(a) + 1:
            continue
        key = (SEED[n], a, b, round(float(po[k - 1][0]), 2))
        if key in seen:
            continue
        seen.add(key)
        i0 = int(np.searchsorted(tu, po[k - 1][0] - 1e-6)); j0 = int(np.searchsorted(tt, ptheir[a] - 1e-6))
        tl = tu[i0] - L; A, TB = a_at(tl), max(BURN - tl, 0.01)
        bp, bv = Bu[i0, b, :3], Bu[i0, b, 3:]
        T_own = reach(Ru[i0, :3], Ru[i0, 3:6], bp, bv, A, TB)
        T_vth = reach(Ru[i0, :3], Rt[j0, 3:6], bp, bv, A, TB)
        T_v0 = reach(Ru[i0, :3], bv, bp, bv, A, TB)  # at rest relative to the balloon
        # their own T0 (their state, their time)
        Bt = th["B"].astype(float)
        T_their = reach(Rt[j0, :3], Rt[j0, 3:6], Bt[j0, b, :3], Bt[j0, b, 3:], a_at(tt[j0] - float(th["launch"])), max(BURN - (tt[j0] - float(th["launch"])), 0.01))
        rows.append(dict(name=n, a=a, b=b, T_own=T_own, T_vtheir=T_vth, T_relrest=T_v0, T_their=T_their,
                         hop_ours=po[k][0] - po[k - 1][0], hop_their=ptheir[b] - ptheir[a],
                         spd_ours=np.linalg.norm(Ru[i0, 3:6]), spd_their=np.linalg.norm(Rt[j0, 3:6]),
                         dt_late=tu[i0] - tt[j0]))
r = pd.DataFrame(rows); r = r[np.isfinite(r[["T_own", "T_vtheir", "T_their"]]).all(1)]
pd.set_option("display.width", 200)
print(f"n pairs {len(r)}")
print(r[["hop_ours", "hop_their", "T_own", "T_vtheir", "T_their", "T_relrest", "spd_ours", "spd_their", "dt_late"]].describe().loc[["mean", "50%"]].round(2).to_string())
print("T0 gap (ours - theirs) mean %.2f; of which velocity (T_own - T_vtheir) %.2f, position/timing (T_vtheir - T_their) %.2f" % (
    (r.T_own - r.T_their).mean(), (r.T_own - r.T_vtheir).mean(), (r.T_vtheir - r.T_their).mean()))
# previous hop stretch vs next T0
s = h[(h.kind == "ours_strict")].drop_duplicates(["seed", "b", "ta", "tb"]).sort_values(["name", "k"])
s["s_prev"] = s.groupby("name").s_mean.shift(1)
s = s.dropna(subset=["s_prev"]); s = s[np.isfinite(s.T1)]
print(s.groupby(pd.cut(s.s_prev, [0, 0.95, 1.05, 1.4]), observed=True).agg(n=("T1", "size"), T0_next=("T1", "mean"), speed0=("speed0", "mean")).round(2).to_string())

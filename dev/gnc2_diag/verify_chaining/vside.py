"""(a) straight-line (net accel = 0) closest approach to b_k from tau before the pop, projected on the next balloon's
direction perpendicular to the approach; (b) the mean net acceleration (dv/dt) over [tau, pop] projected the same way
-> is the 'far side' offset just the signature of accelerating toward the next balloon during the approach?
(c) actual closest approach at the pop and on which side."""
import numpy as np, pandas as pd
from pathlib import Path
import vhops as V
here = Path(__file__).parent
DUP = {"13_Apex", "16_SpaceBerry"}; FREE = {"10_CoolBa", "10_SpaceY", "11_near_miss", "12_Apex", "14_SpaceBerry"}
rows = []
for kind, keep in [("their", set(V.NAMES)), ("ours_strict", set(V.NAMES) - DUP), ("ours_free", FREE)]:
    for n in sorted(keep):
        z = np.load(V.D / f"{kind}_{n}.npz"); t, R, B, S = z["t"], z["R"], z["B"].astype(float), z["S"]
        live = ~np.isnan(R[:, 0]); launch = t[int(np.argmax(live))]
        pops = sorted((int(np.flatnonzero(S[:, b] == 2)[0]), b) for b in range(100) if (S[:, b] == 2).any())
        P = [(i, b) for i, b in pops if t[i] <= launch + 30]
        for k in range(len(P) - 1):
            (i, b), (_, b1) = P[k], P[k + 1]
            prev = P[k - 1][0] if k > 0 else int(np.argmax(live))
            for lag in (30, 50, 80):
                j = i - lag
                if j <= prev: continue
                r, vr = B[j, b, :3] - R[j, :3], B[j, b, 3:] - R[j, 3:6]
                tc = max(-(r @ vr) / (vr @ vr), 0.0); off = -(r + vr * tc)
                u = -vr / np.linalg.norm(vr)  # approach direction (rocket rel. balloon)
                nx = B[j, b1, :3] - B[j, b, :3]; nx -= (nx @ u) * u; nxh = nx / np.linalg.norm(nx)
                # mean relative acceleration of the rocket w.r.t. balloon over [j, i]: from velocity change
                arel = ((R[i, 3:6] - B[i, b, 3:]) - (R[j, 3:6] - B[j, b, 3:])) / (t[i] - t[j])
                # actual rocket-balloon offset at the pop step
                offp = R[i, :3] - B[i, b, :3]
                rows.append(dict(kind=kind, name=n, k=k, lag=lag, off_next=off @ nxh, cos=off @ nxh / max(np.linalg.norm(off), 1e-9),
                                 miss=np.linalg.norm(off), acc_next=arel @ nxh, pred_off=0.5 * (arel @ nxh) * (t[i] - t[j]) ** 2,
                                 pop_off_next=offp @ nxh, pop_dist=np.linalg.norm(offp)))
df = pd.DataFrame(rows)
df.to_csv(here / "vside.csv", index=False)
g = df.groupby(["kind", "lag"])
out = g.agg(n=("cos", "size"), cos_mean=("cos", "mean"), frac_next=("cos", lambda x: (x > 0).mean()), off_next_med=("off_next", "median"),
            miss_med=("miss", "median"), acc_next_med=("acc_next", "median"), pred_off_med=("pred_off", "median"),
            pop_off_next_med=("pop_off_next", "median"), pop_dist_med=("pop_dist", "median"))
pd.set_option("display.width", 250)
print(out.round(2).to_string())

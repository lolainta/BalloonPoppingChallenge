"""Extra measurements:
(1) hit offset: closest distance rocket <-> balloon centre on the pop step (swept segment), their vs ours (aim at the centre
    vs use the 1.5 m radius);
(2) T1 with tol 0 (centre, what our GNC plans) vs tol 1.5 m (the real criterion) at hop start -> time the radius is worth;
(3) balloon drift aim error: constant-velocity prediction of the target position at the pop time from the hop start vs actual;
(4) target switches: strict teacher switches caused by releases, free-mode switches away from a balloon that is not popped
    within 1.5 s (abandoned) and the time spent on abandoned targets."""
import pickle
import numpy as np, pandas as pd
from chases import D, NAMES, reach, a_at, BURN
from hops import SEED

rows, drift, sw = [], [], []
seen = set()
for n in NAMES:
    for kind in ("their", "ours_strict", "ours_free"):
        f = np.load(D / f"{kind}_{n}.npz")
        t, R, B, S, L, pops = f["t"], f["R"], f["B"].astype(float), f["S"], float(f["launch"]), f["pops"]
        for k, (tp, b) in enumerate(pops):
            b = int(b); key = (kind, SEED[n], b, round(float(tp), 2)); dup = key in seen; seen.add(key)
            i = int(np.searchsorted(t, tp - 1e-6))
            # swept closest approach over [i-1, i]
            r0 = B[i - 1, b, :3] - R[i - 1, :3]; r1 = B[i, b, :3] - R[i, :3]; dr = r1 - r0
            s = np.clip(-(r0 @ dr) / max(dr @ dr, 1e-12), 0, 1); dmin = np.linalg.norm(r0 + s * dr)
            row = dict(kind=kind, name=n, dup=dup, k=k, b=b, off=dmin, vrel=np.linalg.norm(dr) / (t[i] - t[i - 1]))
            if k >= 1 and tp <= L + BURN + 0.5:
                ta = pops[k - 1][0]; i0 = int(np.searchsorted(t, ta - 1e-6)); tl = t[i0] - L
                args = (R[i0, :3], R[i0, 3:6], B[i0, b, :3], B[i0, b, 3:], a_at(tl), max(BURN - tl, 0.01))
                row.update(T1c=reach(*args, tol=0.0), T1r=reach(*args, tol=1.5))
                pred = B[i0, b, :3] + B[i0, b, 3:] * (t[i] - t[i0])
                row.update(drift_err=np.linalg.norm(pred - B[i, b, :3]), hop=t[i] - t[i0])
            rows.append(row)
        if kind != "their":
            TGT = f["TGT"]; ptime = {int(b): tp for tp, b in pops}
            ch = np.flatnonzero(np.diff(TGT) != 0)
            for c in ch:
                a_, b_ = int(TGT[c]), int(TGT[c + 1])
                if a_ < 0 or b_ < 0:
                    continue
                tc = t[c + 1]; tpa = ptime.get(a_, np.inf)
                j = c
                while j > 0 and TGT[j - 1] == a_:
                    j -= 1
                sw.append(dict(kind=kind, name=n, seed=SEED[n], t=tc, frm=a_, to=b_, popped_before=tpa <= tc + 0.02,
                               popped_soon=tc < tpa <= tc + 1.5, abandoned=tpa > tc + 1.5, seg=tc - t[j],
                               status_frm=int(S[c + 1, a_])))
r = pd.DataFrame(rows); s = pd.DataFrame(sw)
pd.set_option("display.width", 220)
x = r[~r.dup]
print("(1) hit offset at pop (m): ", x.groupby("kind").off.describe(percentiles=[.25, .5, .75]).round(2).to_string())
print("    vrel at pop (m/s):", x.groupby("kind").vrel.median().round(1).to_dict())
y = x.dropna(subset=["T1c"]); y = y[np.isfinite(y.T1c) & np.isfinite(y.T1r)]
print("(2) T1 centre - T1 radius at hop start (s): mean", y.assign(d=y.T1c - y.T1r).groupby("kind").d.mean().round(3).to_dict(),
      " median", y.assign(d=y.T1c - y.T1r).groupby("kind").d.median().round(3).to_dict())
print("(3) constant-velocity drift error at pop time (m): ", y.groupby("kind").drift_err.describe(percentiles=[.5, .9]).round(3).to_string())
s = s.drop_duplicates(["kind", "seed", "t", "frm", "to"])
print("(4) target switches (dedup):")
for kind in ("ours_strict", "ours_free"):
    z = s[s.kind == kind]
    print(f"  {kind}: switches {len(z)}  after pop {int(z.popped_before.sum())}  pre-pop handoff (popped <=1.5 s later) {int(z.popped_soon.sum())}"
          f"  abandoned {int(z.abandoned.sum())} (time on abandoned targets {z[z.abandoned].seg.sum():.1f} s; of those target still on ground {int((z[z.abandoned].status_frm == 0).sum())})")
pickle.dump((r, s), open("extras.pkl", "wb"))

"""Do they give up current-hop efficiency to set up the next hop? For each hop k (b_k) with a following pop b_{k+1}, in the
middle of the hop (0.5 s after the start to 0.5 s before the pop), the deviation of the actual specific-force direction from
the ideal min-time direction for b_k, d = unit(f) - u_ideal, and its component along the direction b_k -> b_{k+1} (at the
pop of b_k). Positive = steering off the min-time path TOWARD the next balloon. Also the angle between the velocity at the
pop of b_k and the direction to b_{k+1}."""
import numpy as np, pandas as pd
from chases import D, NAMES, reach, a_at, BURN, G
from hops import SEED

rows, seen = [], set()
for n in NAMES:
    for kind in ("their", "ours_strict", "ours_free"):
        f = np.load(D / f"{kind}_{n}.npz")
        t, R, B, L, pops = f["t"], f["R"], f["B"].astype(float), float(f["launch"]), f["pops"]
        acc = np.gradient(R[:, 3:6], t, axis=0) - G
        k9 = np.ones(9) / 9; acc = np.stack([np.convolve(acc[:, c], k9, "same") for c in range(3)], 1)
        for k in range(1, len(pops) - 1):
            ta, tb, b, bn = pops[k - 1][0], pops[k][0], int(pops[k][1]), int(pops[k + 1][1])
            if pops[k + 1][0] > L + BURN + 0.5:
                continue
            key = (kind, SEED[n], b, round(float(tb), 2))
            if key in seen:
                continue
            seen.add(key)
            i1 = int(np.searchsorted(t, tb - 1e-6))
            nxt = B[i1, bn, :3] - B[i1, b, :3]; nxt /= np.linalg.norm(nxt)
            comp, dev = [], []
            for i in range(int(np.searchsorted(t, ta + 0.5)), int(np.searchsorted(t, tb - 0.5)), 5):
                tl = t[i] - L
                T = reach(R[i, :3], R[i, 3:6], B[i, b, :3], B[i, b, 3:], a_at(tl), max(BURN - tl, 0.01))
                if not np.isfinite(T):
                    continue
                m = (B[i, b, :3] - R[i, :3]) + (B[i, b, 3:] - R[i, 3:6]) * T - 0.5 * G * T ** 2
                u = m / np.linalg.norm(m); fa = acc[i] / np.linalg.norm(acc[i])
                comp.append((fa - u) @ nxt); dev.append(np.degrees(np.arccos(np.clip(fa @ u, -1, 1))))
            vhat = R[i1, 3:6] / np.linalg.norm(R[i1, 3:6])
            rows.append(dict(kind=kind, comp=np.mean(comp) if comp else np.nan, dev=np.median(dev) if dev else np.nan,
                             v_next_ang=np.degrees(np.arccos(np.clip(vhat @ nxt, -1, 1)))))
r = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(r.groupby("kind").agg(n=("comp", "count"), dev_median=("dev", "median"), toward_next_mean=("comp", "mean"),
                            frac_toward_next=("comp", lambda x: (x.dropna() > 0).mean()), vel_vs_next_angle_median=("v_next_ang", "median")).round(3).to_string())

"""Pop-to-pop hop table (uniform for their / ours_strict / ours_free): for consecutive pops (t_{k-1}) -> (t_k, b_k):
hop = t_k - t_{k-1}; T1 = ideal min reach time of b_k from the state at t_{k-1} (chases.reach, same a_thrust profile);
excess = hop - T1; miss passes of b_k in between (distance local minima > 1.5 m, < 40 m) and the time lost after the first;
ours: mean stretch over the hop, targeted_at_pop (TGT == b_k right after the previous pop), lead = how long before t_{k-1}
b_k had already become the target (free mode hands off early), vrel at the hit.
-> hops.pkl"""
import pickle
import numpy as np
from pathlib import Path
from chases import D, NAMES, reach, a_at, passes, BURN

OUT = Path(__file__).resolve().parent
SEED = {"10_CoolBa": 0, "10_SpaceY": 14, "11_Mochi": 0, "11_near_miss": 1362, "12_Apex": 1957083, "13_Apex": 1957083, "13_Yaka": 0,
        "14_SpaceBerry": 205, "16_SpaceBerry": 1957083, "16_TeamDJ": 0, "18_Taoyuan": 1957083}


def hops_of(kind, n):
    f = np.load(D / f"{kind}_{n}.npz")
    t, R, B, S, L, pops = f["t"], f["R"], f["B"].astype(float), f["S"], float(f["launch"]), f["pops"]
    rows = []
    for k in range(1, len(pops)):
        ta, tb_ = pops[k - 1][0], pops[k][0]; b = int(pops[k][1])
        if tb_ > L + BURN + 0.5:  # coasting pop after burnout: not a guided hop
            continue
        i0 = int(np.searchsorted(t, ta - 1e-6)); i1 = int(np.searchsorted(t, tb_ - 1e-6))
        tl = t[i0] - L
        T1 = reach(R[i0, :3], R[i0, 3:6], B[i0, b, :3], B[i0, b, 3:], a_at(tl), max(BURN - tl, 0.01))
        d = np.linalg.norm(B[:, b, :3] - R[:, :3], axis=1)
        ps = passes(t, d, i0, i1); miss = [(x, dm) for x, dm in ps if dm > 1.5]
        vr = R[i1, 3:6] - B[i1, b, 3:]
        row = dict(kind=kind, name=n, seed=SEED[n], k=k, b=b, bprev=int(pops[k - 1][1]), ta=ta, tb=tb_, tl=tl, hop=tb_ - ta, T1=T1,
                   d0=d[i0], nmiss=len(miss), miss1=miss[0][1] if miss else np.nan, lost=(tb_ - miss[0][0]) if miss else 0.0,
                   vrel=np.linalg.norm(vr), speed0=np.linalg.norm(R[i0, 3:6]))
        if kind != "their":
            TGT, ST = f["TGT"], f["STRETCH"]
            seg = TGT[i0:i1 + 1]
            row["targeted_at_pop"] = bool(TGT[min(i0 + 1, len(t) - 1)] == b)
            j = i0
            while j > 0 and TGT[j - 1] == b:
                j -= 1
            row["lead"] = t[i0] - t[j] if TGT[i0] == b else 0.0
            row["frac_on_b"] = float((seg == b).mean())
            m = seg >= 0
            row["s_mean"] = float(ST[i0:i1 + 1][m].mean()) if m.any() else np.nan
            row["nswitch"] = int((np.diff(seg) != 0).sum())
        rows.append(row)
    return rows


if __name__ == "__main__":
    rows = []
    for n in NAMES:
        for kind in ("their", "ours_strict", "ours_free"):
            rows += hops_of(kind, n)
    pickle.dump(rows, open(OUT / "hops.pkl", "wb"))
    import pandas as pd
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    h = pd.DataFrame(rows); h["excess"] = h.hop - h.T1
    for lab, df in (("ALL", h), ("DEDUP", h.drop_duplicates(["kind", "seed", "b", "ta", "tb"]))):
        print("====", lab)
        for kind in ("their", "ours_strict", "ours_free"):
            x = df[(df.kind == kind) & np.isfinite(df.T1)]
            c = x[x.nmiss == 0]
            print(f"{kind:12s} n {len(x):3d} hop med {x.hop.median():.2f} mean {x.hop.mean():.2f} | T1 med {x.T1.median():.2f} mean {x.T1.mean():.2f} | "
                  f"excess med {x.excess.median():.2f} mean {x.excess.mean():.2f} | clean(n={len(c)}) hop {c.hop.mean():.2f} T1 {c.T1.mean():.2f} excess {c.excess.mean():.2f} | "
                  f"miss-pass {int((x.nmiss > 0).sum())} ({100 * (x.nmiss > 0).mean():.0f}%) lost/hop {x.lost.mean():.2f} | vrel med {x.vrel.median():.1f} | speed0 med {x.speed0.median():.1f} | d0 med {x.d0.median():.1f}")

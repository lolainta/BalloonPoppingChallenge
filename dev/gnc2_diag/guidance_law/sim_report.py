"""Report on sims/<variant>__<name>.npz vs the logged baseline (data/gnc2/ours_strict_*): pops, pop-to-pop hop stats
(mean/median, T1 at hop start, excess), near-miss re-passes (distance local min 1.5-6 m while targeted >= 0.8 s),
hit offset (aim), launch -> first pop. Dedup = the 7 distinct (seed, order) flights' identical hops counted once.
.venv/bin/python dev/gnc2_diag/guidance_law/sim_report.py [variant ...]"""
import sys
import numpy as np, pandas as pd
from pathlib import Path
from chases import D, NAMES, reach, a_at, BURN
from hops import SEED

SIMS = Path(__file__).resolve().parent / "sims"


FREE_NAMES = ["10_CoolBa", "10_SpaceY", "11_near_miss", "12_Apex", "14_SpaceBerry"]  # the 5 distinct seeds (free runs repeat per seed)


def load(v, n):
    if v == "base":
        return np.load(D / f"ours_strict_{n}.npz")
    if v == "fbase":
        return np.load(D / f"ours_free_{n}.npz")
    return np.load(SIMS / f"{v}__{n}.npz")


def hops(v, n):
    f = load(v, n)
    t, R, B, L, pops = f["t"], f["R"], f["B"].astype(float), float(f["launch"]), f["pops"]
    TGT, ST = f["TGT"], f["STRETCH"]
    out = []
    for k in range(1, len(pops)):
        ta, tb, b = pops[k - 1][0], pops[k][0], int(pops[k][1])
        if tb > L + BURN + 0.5:
            continue
        i0 = int(np.searchsorted(t, ta - 1e-6)); i1 = int(np.searchsorted(t, tb - 1e-6)); tl = t[i0] - L
        T1 = reach(R[i0, :3], R[i0, 3:6], B[i0, b, :3], B[i0, b, 3:], a_at(tl), max(BURN - tl, 0.01))
        d = np.linalg.norm(B[i0:i1 + 1, b, :3] - R[i0:i1 + 1, :3], axis=1)
        lm = np.flatnonzero((d[1:-1] < d[:-2]) & (d[1:-1] <= d[2:])) + 1
        nm = [j for j in lm if 1.5 < d[j] < 6.0 and j < len(d) - 2 and (TGT[i0:i0 + j] == b).sum() >= 80]
        j2 = min(i1 + 40, len(t) - 1)
        bc = B[i1, b, :3][None] + B[i1, b, 3:][None] * (t[i1 - 5:j2 + 1] - t[i1])[:, None]
        off = np.linalg.norm(R[i1 - 5:j2 + 1, :3] - bc, axis=1).min() if not np.isnan(R[j2, 0]) else np.nan
        seg = TGT[i0:i1 + 1]
        Lx = {}
        for dt_ in (0.5, 1.0):
            j = int(np.searchsorted(t, ta + dt_ - 1e-6))
            if j < i1:
                Tj = reach(R[j, :3], R[j, 3:6], B[j, b, :3], B[j, b, 3:], a_at(t[j] - L), max(BURN - (t[j] - L), 0.01))
                Lx[dt_] = dt_ + Tj - T1
            else:
                Lx[dt_] = np.nan
        m = (B[i0, b, :3] - R[i0, :3]) + (B[i0, b, 3:] - R[i0, 3:6]) * T1 + 0.5 * 9.81 * np.array([0, 0, 1.0]) * T1 ** 2
        e0, e1, e2, e3 = R[i0, 6:10]
        z = np.array([2 * (e1 * e3 + e0 * e2), 2 * (e2 * e3 - e0 * e1), 1 - 2 * (e1 * e1 + e2 * e2)])
        ang0 = np.degrees(np.arccos(np.clip(z @ m / max(np.linalg.norm(m), 1e-9), -1, 1))) if np.isfinite(T1) else np.nan
        out.append(dict(L05=Lx[0.5], L10=Lx[1.0], ang0=ang0, v=v, name=n, seed=SEED[n], k=k, b=b, ta=ta, tb=tb, hop=tb - ta, T1=T1, near=len(nm),
                        lost=(tb - t[i0 + nm[0]]) if nm else 0.0, off=off, s=float(ST[i0:i1 + 1][seg >= 0].mean()) if (seg >= 0).any() else np.nan,
                        vrel=np.linalg.norm(R[i1, 3:6] - B[i1, b, 3:])))
    return out, len(pops), (pops[0][0] - L) if len(pops) else np.nan, sum(1 for p in pops if p[0] <= L + BURN + 0.5)


def report(vs, names=None):
    global NAMES
    NAMES = names or NAMES
    res = []
    for v in vs:
        H, tot, first, inburn = [], 0, [], 0
        for n in NAMES:
            try:
                h, c, fp, ib = hops(v, n)
            except FileNotFoundError:
                break
            H += h; tot += c; first.append(fp); inburn += ib
        else:
            h = pd.DataFrame(H); dd = h.drop_duplicates(["seed", "b", "ta", "tb"])
            per = [len(load(v, n)["pops"]) for n in NAMES]
            fin = dd[np.isfinite(dd.T1)]
            res.append(dict(variant=v, pops=tot, per=" ".join(map(str, per)), hop_mean=h.hop.mean(), hop_med=h.hop.median(),
                            dd_n=len(dd), dd_hop_mean=dd.hop.mean(), dd_T1=fin.T1.mean(), dd_excess=(fin.hop - fin.T1).mean(),
                            near=int(dd.near.sum()), lost_per_hop=dd.lost.sum() / len(dd), off_med=dd.off.median(), vrel=dd.vrel.median(),
                            s=dd.s.mean(), first_med=np.nanmedian(first), L05=fin.L05.mean(), L10=fin.L10.mean(), ang0=fin.ang0.median()))
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(pd.DataFrame(res).round(2).to_string(index=False))


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "--free":
        report(a[1:] or ["fbase"], FREE_NAMES)
    else:
        report(a or ["base"])

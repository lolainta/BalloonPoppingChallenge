"""Hop-start gap decomposition: CHOICE (which balloon is next) vs POSITION (how many balloons are in reach).

At every pop k (hop start; state = last log step before balloon b_k's status turns 2), for pop-to-pop hops inside the
burn (launch -> first pop excluded), reach-time model = dev/gnc2_diag/hop_geometry/hops.py tmin() (the §132 model):
constant specific force |f| <= a_cap (95th pct of the logged thrust accel during the burn, per flight), gravity, balloon
constant velocity, hit radius 1.5 m, no drag / coast / drift. Per hop:
  T_ch      Tmin to the balloon actually popped next
  T_b1..5   1st/2nd/3rd/5th smallest Tmin over the alive (status 1) balloons, the just-popped one excluded
  n10/15/20 alive balloons with Tmin < 1.0 / 1.5 / 2.0 s
  rank      rank of the chosen next among alive balloons by Tmin (1 = best); nan if it was not flying yet
  alt, spd  altitude above the pad (z - 20) and speed at the pop
  ang_c     angle between the rocket velocity and the direction to the centroid of the intercept points of the alive
            balloons with Tmin <= 2.0 s (top 3 if fewer than 2) -- "densest reachable cluster" (a)
  ang_d     angle between velocity and the direction to the densest cluster (b): among alive balloons with Tmin < 3 s,
            the one with most alive neighbours within 15 m; centroid of it + those neighbours
.venv/bin/python dev/gnc2_seq/decomp/decomp.py   -> hops_s1.csv, hops_s4.csv + printed tables
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "dev/gnc2_diag/hop_geometry"))
from hops import thrust_acc, G, R_HIT  # noqa: E402

GRID = np.arange(0.01, 15.0, 0.01)
TCAP = 15.0
S1 = ROOT / "data/gnc2"
NAMES = "10_CoolBa 10_SpaceY 11_Mochi 11_near_miss 12_Apex 13_Apex 13_Yaka 14_SpaceBerry 16_SpaceBerry 16_TeamDJ 18_Taoyuan".split()
FREE_KEEP = {"10_CoolBa", "10_SpaceY", "11_near_miss", "12_Apex", "14_SpaceBerry"}  # one ours_free flight per seed


def tmin_all(p, v, bp, bv, a):
    """(N,) min reach time of every balloon (hops.tmin vectorised); TCAP if unreachable within the grid."""
    tg = GRID[:, None, None]
    miss = (bp - p)[None] + (bv - v)[None] * tg - 0.5 * G * tg ** 2
    ok = np.linalg.norm(miss, axis=2) <= a * 0.5 * GRID[:, None] ** 2 + R_HIT
    return np.where(ok.any(0), GRID[np.argmax(ok, 0)], TCAP)


def ang(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return np.nan if na < 1e-6 or nb < 1e-6 else float(np.degrees(np.arccos(np.clip(a @ b / (na * nb), -1, 1))))


def flight(path, burn=30.0):
    z = np.load(path)
    t, R, B, S, pops = z["t"], z["R"], z["B"], z["S"], z["pops"]
    launch = float(z["launch"])
    live = ~np.isnan(R[:, 0]); idx = np.flatnonzero(live)
    fa = np.full_like(R[:, :3], np.nan); fa[idx] = thrust_acc(t[idx], R[idx])
    fmag = np.linalg.norm(fa, axis=1)
    a_cap = float(np.nanpercentile(fmag[live & (t > launch + 0.5) & (t < launch + burn - 0.5)], 95))
    P = [(float(tp), int(b)) for tp, b in pops if tp <= launch + burn]
    rows = []
    for k in range(len(P) - 1):
        (t0, b0), (t1, b1) = P[k], P[k + 1]
        j0 = int(np.argmax(S[:, b0] == 2)) - 1
        p, v = R[j0, :3], R[j0, 3:6]
        st = S[j0].copy(); st[b0] = 2
        al = np.flatnonzero(st == 1)
        tm = tmin_all(p, v, B[j0, al, :3].astype(float), B[j0, al, 3:6].astype(float), a_cap)
        o = np.sort(tm)
        pick = lambda i: o[i] if len(o) > i else TCAP
        tch = float(tmin_all(p, v, B[j0, [b1], :3].astype(float), B[j0, [b1], 3:6].astype(float), a_cap)[0])
        rank = float(1 + (tm < tch - 1e-9).sum()) if S[j0, b1] == 1 else np.nan
        bp, bv = B[j0, al, :3].astype(float), B[j0, al, 3:6].astype(float)
        ip = bp + bv * np.minimum(tm, 5.0)[:, None]  # intercept points
        sel = tm <= 2.0
        if sel.sum() < 2:
            sel = np.zeros(len(tm), bool); sel[np.argsort(tm)[:3]] = True
        ang_c = ang(v, ip[sel].mean(0) - p)
        cand = np.flatnonzero(tm < 3.0)
        if len(cand):
            D = np.linalg.norm(bp[cand, None] - bp[None], axis=2) < 15.0
            c = cand[np.argmax(D.sum(1))]; nb = np.linalg.norm(bp - bp[c], axis=1) < 15.0
            ang_d, n_d = ang(v, bp[nb].mean(0) - p), int(nb.sum())
        else:
            ang_d, n_d = np.nan, 0
        rows.append(dict(t0=t0, T=t1 - t0, tb_left=launch + burn - t0, b0=b0, b1=b1, st_b1=int(S[j0, b1]), n_alive=len(al),
                         T_ch=tch, T_b1=pick(0), T_b2=pick(1), T_b3=pick(2), T_b5=pick(4), excess=tch - pick(0),
                         n10=int((tm < 1.0).sum()), n15=int((tm < 1.5).sum()), n20=int((tm < 2.0).sum()), rank=rank,
                         alt=p[2] - 20.0, spd=float(np.linalg.norm(v)), vz=v[2], ang_c=ang_c, ang_d=ang_d, n_dense=n_d,
                         ang_ch=ang(v, B[j0, b1, :3] + B[j0, b1, 3:6] * min(tch, 5.0) - p), a_cap=a_cap))
    return rows


def boot(df, col, f, B=5000, rng=np.random.default_rng(0)):
    """statistic f over hops pooled from resampled seeds -> (point, lo, hi) 95%."""
    g = {s: d[col].to_numpy(float) for s, d in df.groupby("seed")}
    keys = list(g); pt = f(np.concatenate([g[k] for k in keys]))
    bs = [f(np.concatenate([g[k] for k in rng.choice(keys, len(keys))])) for _ in range(B)]
    return pt, *np.percentile(bs, [2.5, 97.5])


def pboot(a, b, col, B=5000, rng=np.random.default_rng(1)):
    """paired over seeds: mean over seeds of (seed mean ours - seed mean theirs)."""
    ma, mb = a.groupby("seed")[col].mean(), b.groupby("seed")[col].mean()
    d = (ma - mb).dropna().to_numpy()
    bs = [rng.choice(d, len(d)).mean() for _ in range(B)]
    return d.mean(), *np.percentile(bs, [2.5, 97.5])


COLS = ["T_ch", "T_b1", "T_b2", "T_b3", "T_b5", "excess", "n10", "n15", "n20", "rank", "alt", "spd", "ang_c", "ang_d", "ang_ch", "T"]


def table(groups, ref=None):
    for col in COLS:
        line = f"{col:7s}"
        for name, df in groups:
            m, ml, mh = boot(df, col, np.nanmedian); a, al, ah = boot(df, col, np.nanmean)
            line += f" | {name}: med {m:6.2f} [{ml:5.2f},{mh:5.2f}] mean {a:6.2f} [{al:5.2f},{ah:5.2f}]"
        if ref:
            d, lo, hi = pboot(ref[0], ref[1], col)
            line += f" | paired seed-mean diff {d:+.2f} [{lo:+.2f},{hi:+.2f}]"
        print(line)


def extra(name, df):
    r = df["rank"]
    print(f"{name}: hops {len(df)}, seeds {df.seed.nunique()}; chosen rank==1 {np.mean(r == 1):.2f}, <=2 {np.mean(r <= 2):.2f}, "
          f"<=3 {np.mean(r <= 3):.2f}; chosen not flying yet {np.mean(df.st_b1 != 1):.2f}; "
          f"excess>0.3 s {np.mean(df.excess > 0.3):.2f}; share of mean T_ch that is excess {df.excess.mean() / df.T_ch.mean():.2f}")


if __name__ == "__main__":
    th, fr = [], []
    for n in NAMES:
        for r in flight(S1 / f"their_{n}.npz"):
            th.append(dict(r, name=n))
        if n in FREE_KEEP:
            for r in flight(S1 / f"ours_free_{n}.npz"):
                fr.append(dict(r, name=n))
    seeds = {n: int(np.load(S1 / f"their_{n}.npz")["seed"]) for n in NAMES}
    th = pd.DataFrame(th); fr = pd.DataFrame(fr)
    th["seed"] = th.name.map(seeds); fr["seed"] = fr.name.map(seeds)
    th["kind"] = "their"; fr["kind"] = "ours_free"
    pd.concat([th, fr]).to_csv(HERE / "hops_s1.csv", index=False)
    print("=== scenario 1 (5 distinct seeds; theirs = 11 replays pooled per seed; ours_free = 1 flight per seed) ===")
    print("bootstrap 95% over seeds (5000 resamples); paired diff = ours - theirs of per-seed means")
    table([("ours", fr), ("their", th)], ref=(fr, th))
    extra("ours_free", fr); extra("their", th)
    # subset: hops with >= 3 s burn left (end-of-burn hops excluded)
    print("--- tb_left >= 3 s only ---")
    table([("ours", fr[fr.tb_left >= 3]), ("their", th[th.tb_left >= 3])], ref=(fr[fr.tb_left >= 3], th[th.tb_left >= 3]))
    s4 = []
    for f in sorted((HERE / "data").glob("s4free_*.npz")):
        z = np.load(f)
        for r in flight(f, float(z["burn_time"])):
            s4.append(dict(r, seed=int(z["seed"])))
    s4 = pd.DataFrame(s4); s4["kind"] = "s4_free"; s4.to_csv(HERE / "hops_s4.csv", index=False)
    print(f"=== scenario 4 ({s4.seed.nunique()} seeds 36000-36039, current policy + eob=2 tau3 T4) ===")
    table([("s4", s4)])
    extra("s4_free", s4)

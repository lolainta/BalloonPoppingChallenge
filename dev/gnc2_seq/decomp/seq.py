"""Is the non-greedy choice justified by lookahead?  At every hop start (same states / model as decomp.py), chain the
constant-force min-time model: arriving at balloon i at t_i with v_i = v + 2 miss_i / t_i + G t_i (the constant-force
solution's end velocity), balloons moving on (constant velocity), the just-hit one removed. For depth D = 1, 2, 3:
  C_D(first = i) = min over the next D-1 balloons of the summed min reach times (exhaustive over the K=8 best at each level)
  xs_D = C_D(chosen next) - min_i C_D(i)      (lookahead regret of the actual choice)
  gr_D = C_D(greedy best-Tmin balloon) - min_i C_D(i)  (regret of pure greedy)
C_D per hop is also divided by D -> per-hop "sequence Tmin". 
.venv/bin/python dev/gnc2_seq/decomp/seq.py  -> seq_s1.csv, seq_s4.csv + printed tables
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from decomp import G, GRID, R_HIT, TCAP, S1, NAMES, FREE_KEEP, thrust_acc, boot, pboot  # noqa: E402

K = 8


def reach(p, v, bp, bv, a):
    """-> t (N,), end velocity (N, 3)."""
    tg = GRID[:, None, None]
    miss = (bp - p)[None] + (bv - v)[None] * tg - 0.5 * G * tg ** 2
    ok = np.linalg.norm(miss, axis=2) <= a * 0.5 * GRID[:, None] ** 2 + R_HIT
    i = np.argmax(ok, 0); t = np.where(ok.any(0), GRID[i], TCAP)
    m = miss[i, np.arange(len(bp))]
    return t, v + 2 * m / t[:, None] + G * t[:, None]


def best(p, v, bp, bv, alive, a, D):
    """min summed reach time of D more balloons from (p, v) over the alive set (indices into bp)."""
    if D == 0 or len(alive) == 0:
        return 0.0
    t, va = reach(p, v, bp[alive], bv[alive], a)
    out = np.inf
    for j in np.argsort(t)[:K]:
        rest = np.delete(alive, j)
        c = t[j] + best(bp[alive[j]] + bv[alive[j]] * t[j], va[j], bp + bv * t[j], bv, rest, a, D - 1)
        out = min(out, c)
    return out


def costs(p, v, bp, bv, alive, a, D, ci):
    """C_D(first = i) for every alive i among the K best first steps + the actual choice ci (TCAP * D for the rest)."""
    t, va = reach(p, v, bp[alive], bv[alive], a)
    C = np.full(len(alive), TCAP * D)
    for j in set(np.argsort(t)[:K].tolist()) | {ci}:
        rest = np.delete(alive, j)
        C[j] = t[j] + best(bp[alive[j]] + bv[alive[j]] * t[j], va[j], bp + bv * t[j], bv, rest, a, D - 1)
    return t, C


def flight(path, burn=30.0):
    z = np.load(path)
    t, R, B, S, pops = z["t"], z["R"], z["B"], z["S"], z["pops"]
    launch = float(z["launch"]); live = ~np.isnan(R[:, 0]); idx = np.flatnonzero(live)
    fa = np.full_like(R[:, :3], np.nan); fa[idx] = thrust_acc(t[idx], R[idx])
    fmag = np.linalg.norm(fa, axis=1)
    a = float(np.nanpercentile(fmag[live & (t > launch + 0.5) & (t < launch + burn - 0.5)], 95))
    P = [(float(tp), int(b)) for tp, b in pops if tp <= launch + burn]
    rows = []
    for k in range(len(P) - 1):
        b0, b1 = P[k][1], P[k + 1][1]
        j0 = int(np.argmax(S[:, b0] == 2)) - 1
        p, v = R[j0, :3], R[j0, 3:6]
        st = S[j0].copy(); st[b0] = 2
        alive = np.flatnonzero(st == 1)
        if S[j0, b1] != 1:
            continue
        bp, bv = B[j0, :, :3].astype(float), B[j0, :, 3:6].astype(float)
        ci = int(np.flatnonzero(alive == b1)[0])
        r = dict(tb_left=launch + burn - P[k][0])
        for D in (1, 2, 3):
            t1, C = costs(p, v, bp, bv, alive, a, D, ci)
            g = int(np.argmin(t1))
            r.update({f"C{D}_ch": C[ci], f"C{D}_opt": C.min(), f"xs{D}": C[ci] - C.min(), f"gr{D}": C[g] - C.min(),
                      f"opt{D}_is_ch": float(np.argmin(C) == ci), f"opt{D}_is_greedy": float(np.argmin(C) == g)})
        rows.append(r)
    return rows


def show(name, df, ref=None):
    print(f"--- {name}: {len(df)} hops, {df.seed.nunique()} seeds")
    for D in (1, 2, 3):
        for col in (f"xs{D}", f"gr{D}"):
            m, ml, mh = boot(df, col, np.nanmedian); mu, lo, hi = boot(df, col, np.nanmean)
            print(f"  {col}: med {m:.2f} [{ml:.2f},{mh:.2f}] mean {mu:.2f} [{lo:.2f},{hi:.2f}]  (per hop of the {D}-chain: {mu / D:.2f} s)")
        print(f"  D={D}: optimal first = actual choice {df[f'opt{D}_is_ch'].mean():.2f}, = greedy {df[f'opt{D}_is_greedy'].mean():.2f}; "
              f"C{D}_opt / {D} mean {df[f'C{D}_opt'].mean() / D:.2f}, C{D}_ch / {D} mean {df[f'C{D}_ch'].mean() / D:.2f}")


if __name__ == "__main__":
    import multiprocessing as mp
    jobs = [("their", n, S1 / f"their_{n}.npz", 30.0) for n in NAMES] + \
           [("ours_free", n, S1 / f"ours_free_{n}.npz", 30.0) for n in NAMES if n in FREE_KEEP]
    for f in sorted((HERE / "data").glob("s4free_*.npz")):
        jobs.append(("s4_free", f.stem, f, float(np.load(f)["burn_time"])))
    seeds = {j[1]: int(np.load(j[2])["seed"]) for j in jobs}
    with mp.get_context("spawn").Pool(12) as pool:
        res = pool.starmap(flight, [(j[2], j[3]) for j in jobs])
    rows = [dict(r, kind=j[0], name=j[1], seed=seeds[j[1]]) for j, rs in zip(jobs, res) for r in rs]
    df = pd.DataFrame(rows)
    df[df.kind != "s4_free"].to_csv(HERE / "seq_s1.csv", index=False); df[df.kind == "s4_free"].to_csv(HERE / "seq_s4.csv", index=False)
    fr, th, s4 = (df[df.kind == k] for k in ("ours_free", "their", "s4_free"))
    print("scenario 1 (5 distinct seeds), bootstrap 95% over seeds")
    show("ours_free s1", fr); show("their s1", th)
    for D in (1, 2, 3):
        d, lo, hi = pboot(fr, th, f"xs{D}"); print(f"  paired seed-mean xs{D} ours - their {d:+.2f} [{lo:+.2f},{hi:+.2f}]")
        d, lo, hi = pboot(fr, th, f"C{D}_opt"); print(f"  paired seed-mean C{D}_opt ours - their {d:+.2f} [{lo:+.2f},{hi:+.2f}]")
    show("ours_free s4 (40 seeds)", s4)

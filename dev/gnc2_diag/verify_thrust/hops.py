"""Verifier: per-hop point-mass bounds (own implementation) + a linearized magnitude-cost estimate.
T_lb : min T with |b(T) - p0 - v0 T - G T^2/2| <= Dfull(T) + rh, Dfull(T) = int_0^T (T - s) a_max(s) ds
T_mag: same with the realized AXIAL thrust accel (f_sp . nose + axial drag)
dt_lin: extra displacement the unused thrust would have given along the realized thrust directions, projected on the
        approach direction at the hit, / closing speed at the hit  (first-order time the magnitude deficit costs).
"""
import sys, json; sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
import numpy as np
from common import *
from util import run
from breakdown import pops_from_S

def D_of(acc):
    """D[k] = int_0^{k DT} (k DT - s) acc(s) ds, acc piecewise constant on steps (acc[j] on (j DT, (j+1) DT])."""
    n = len(acc)
    D = np.zeros(n + 1)
    c1 = np.concatenate([[0.0], np.cumsum(acc) * DT])  # velocity-like
    # exact for piecewise-constant: D[k+1] = D[k] + c1[k] DT + acc[k] DT^2 / 2
    for k in range(n):
        D[k + 1] = D[k] + c1[k] * DT + 0.5 * acc[k] * DT * DT
    return D

def hops(kind, name, sims=False, rh_fixed=None):
    r = run(kind, name, sims); d, x = r["d"], r["x"]
    P = pops_from_S(d)
    t, p, v, B = x["t"], x["p"], x["v"], d["B"]
    iL = int(np.flatnonzero(np.isfinite(d["R"][:, 0]))[0])
    ev = [(t[iL], -1, iL)] + P
    am, thr = r["am_all"], r["x"]["thr_ax"]
    out = []
    for j, ((t0, _, i0), (t1, b, i1)) in enumerate(zip(ev[:-1], ev[1:])):
        n = i1 - i0
        if n < 2: continue
        tau0 = t[i0] - x["L"]
        if tau0 >= BURN - 0.05: break
        taum = x["taum"][i0 + 1:i1 + 1]
        burning = taum < BURN
        a_full = np.where(burning, am[i0 + 1:i1 + 1], 0.0)
        a_real = np.where(burning & np.isfinite(thr[i0 + 1:i1 + 1]), np.clip(thr[i0 + 1:i1 + 1], 0, None), 0.0)
        a_full = np.maximum(a_full, a_real)
        T = np.arange(n + 1) * DT
        bt = B[i0:i1 + 1, b, :3].astype(float)
        miss = np.linalg.norm(bt - p[i0] - v[i0] * T[:, None] - 0.5 * G * T[:, None] ** 2, axis=1)
        rh = float(np.linalg.norm(p[i1] - B[i1, b, :3])) if rh_fixed is None else rh_fixed
        Df, Dm = D_of(a_full), D_of(a_real)
        okf, okm = miss <= Df + rh, miss <= Dm + rh
        T_lb = T[np.argmax(okf)] if okf.any() else np.nan
        T_mag = T[np.argmax(okm)] if okm.any() else np.nan
        # linearized cost of the magnitude deficit
        vrel = v[i1] - B[i1, b, 3:].astype(float); vc = np.linalg.norm(vrel); u = vrel / max(vc, 1e-6)
        nm = x["nmid"][i0 + 1:i1 + 1]
        proj = np.where(np.isfinite(nm).all(1), nm @ u, 0.0)
        deficit = (a_full - a_real) * proj
        w = (n * DT - (np.arange(n) + 0.5) * DT)
        dD = np.sum(w * deficit) * DT
        out.append(dict(kind=kind, name=name, first=(j == 0), tau0=tau0, T_act=n * DT, T_lb=T_lb, T_mag=T_mag, rh=rh,
                        burn_frac=float(burning.mean()), U_hop=float(np.mean(a_real[burning] / np.where(burning, am[i0 + 1:i1 + 1], 1)[burning])) if burning.any() else np.nan,
                        dD=dD, vc=vc, dt_lin=dD / max(vc, 1.0)))
    return out

def dedup_names(kind, sims=False):
    seen, keep = set(), []
    for n in NAMES:
        d = load(kind, n, sims); key = round(float(np.nansum(d["R"][:, :3])), 3)
        if key not in seen: seen.add(key); keep.append(n)
    return keep

def summarize(H, lab):
    a = lambda k: np.array([h[k] for h in H], float)
    ml, dl = a("T_mag") - a("T_lb"), a("T_act") - a("T_mag")
    return (f"{lab:34s} n={len(H):3d} T_act {a('T_act').mean():.2f}/{np.median(a('T_act')):.2f} T_lb {np.nanmean(a('T_lb')):.2f}/{np.nanmedian(a('T_lb')):.2f} "
            f"magL {np.nanmean(ml):.3f}/{np.nanmedian(ml):.3f} dirL {np.nanmean(dl):.2f}/{np.nanmedian(dl):.2f} U_hop {np.nanmean(a('U_hop')):.3f} "
            f"dt_lin {np.nanmean(a('dt_lin')):.3f}/{np.nanmedian(a('dt_lin')):.3f} vc {a('vc').mean():.1f}  (mean/median)")

if __name__ == "__main__":
    lines = []
    allH = {}
    for kind in ("their", "ours_strict", "ours_free"):
        H = sum((hops(kind, n) for n in NAMES), [])
        allH[kind] = H
        for sel, lab in ((lambda h: not h["first"] and h["burn_frac"] > 0.5, "non-first, burn_frac>0.5"),
                         (lambda h: not h["first"] and h["burn_frac"] >= 1.0, "non-first, fully in burn"),
                         (lambda h: h["first"] and h["burn_frac"] > 0.5, "first")):
            lines.append(summarize([h for h in H if sel(h)], f"{kind} {lab}"))
        dn = dedup_names(kind)
        lines.append(summarize([h for h in H if (not h["first"]) and h["burn_frac"] > 0.5 and h["name"] in dn], f"{kind} non-first, distinct flights({len(dn)})"))
    # fixed hit radius sensitivity
    for kind in ("their", "ours_strict"):
        H = sum((hops(kind, n, rh_fixed=1.5) for n in NAMES), [])
        lines.append(summarize([h for h in H if not h["first"] and h["burn_frac"] > 0.5], f"{kind} non-first rh=1.5 fixed"))
    txt = "\n".join(lines); print(txt)
    (ROOT / "dev/gnc2_diag/verify_thrust/hops.txt").write_text(txt + "\n")
    json.dump({k: v for k, v in allH.items()}, open(ROOT / "dev/gnc2_diag/verify_thrust/hops.json", "w"), default=float)

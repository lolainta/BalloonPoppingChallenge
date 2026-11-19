"""Thrust-magnitude utilization diagnosis (GNC v2, hypothesis: wasted impulse).

Offline, from data/gnc2/*.npz (dev/gnc2_log.py). Mass flow in ActiveRocketPy is a function of time only
(flight.py: mass_flow_rate.get_value_opt(t); throttle only scales thrust), so any throttle < 1 is lost impulse.

Per run, during the burn (0.2 s < tau < 29.95 s after launch):
  f_sp  = dv/dt - G                      specific force (world), central differences of the logged velocity
  f_thr = f_sp - drag_est                thrust accel, drag_est = -k rho |v| v / m(tau)
  a_max = T / m(tau)                     full-throttle thrust accel (validated on our throttle == 1 samples)
  U     = |f_thr| / a_max                utilization
Per hop (previous pop or launch -> next pop, hop start inside the burn), exact point-mass bounds from the hop start state
and the LOGGED balloon trajectory (constant-direction thrust is the max-displacement control, so these are true bounds):
  T_lb  = min T with |b(T) - p0 - v0 T - G T^2/2| <= int_0^T (T - s) a_max(s) ds + r_hit      (full thrust)
  T_mag = same with the realized |f_sp(s)| instead of a_max(s)                                   (realized magnitude)
  T_act = actual hop time.  T_mag - T_lb: time lost to thrust magnitude; T_act - T_mag: lost to direction/path.

.venv/bin/python dev/gnc2_diag/thrust_util/util_analysis.py
"""
import glob, json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
G = np.array([0.0, 0.0, -9.81])
DT = 0.01
THRUST = 1080.0
M0 = 70 + 13.9 + 2.13 + np.pi * (0.0843**2 - 0.0295**2) * 0.2757 * 900  # body + liquid + gas + grain
MDOT = 0.425 + np.pi * (0.0843**2 - 0.0295**2) * 0.2757 * 900 / 30.0  # liquid out + grain over the 30 s burn
K_DRAG = 0.5 * 0.25 * np.pi * 0.22**2  # 0.5 Cd A
BURN = 30.0
NAMES = "10_CoolBa 10_SpaceY 11_Mochi 11_near_miss 12_Apex 13_Apex 13_Yaka 14_SpaceBerry 16_SpaceBerry 16_TeamDJ 18_Taoyuan".split()
KINDS = ("their", "ours_strict", "ours_free")
STRETCH_LEVELS = (0.6, 0.8, 1.0, 1.3)


A_CAL = 1.0097  # median |f_thr| / (T / m(tau)) on our throttle == 1 samples (checks.py): pressure thrust / drag model error


def a_max(tau):
    tau = np.clip(tau, 0.0, BURN)
    return A_CAL * THRUST / (M0 - MDOT * tau)


def mass(tau):
    return M0 - MDOT * np.clip(tau, 0.0, BURN)


def load(kind, name):
    d = np.load(ROOT / f"data/gnc2/{kind}_{name}.npz")
    return {k: d[k] for k in d.files}


def derive(d):
    t, R = d["t"], d["R"]
    L = float(d["launch"])
    tau = t - L
    v = R[:, 3:6]
    dv = np.full_like(v, np.nan)
    dv[1:-1] = (v[2:] - v[:-2]) / (2 * DT)
    f_sp = dv - G
    alt = R[:, 2] - 20.0
    rho = 1.225 * np.exp(-np.maximum(alt + 20.0, 0.0) / 8500.0)
    sp = np.linalg.norm(v, axis=1, keepdims=True)
    drag = -K_DRAG * rho[:, None] * sp * v / mass(tau)[:, None]
    f_thr = f_sp - drag
    am = a_max(tau)
    burn = (tau > 0.2) & (tau < BURN - 0.05) & np.isfinite(f_sp).all(1) & np.isfinite(R[:, 0])
    return dict(t=t, tau=tau, L=L, p=R[:, :3], v=v, f_sp=f_sp, f_thr=f_thr, drag=drag, am=am, burn=burn,
                U=np.linalg.norm(f_thr, axis=1) / am, Usp=np.linalg.norm(f_sp, axis=1) / am)


def targets_of(d, x):
    """Current target per step: the next balloon popped (their order for 'their'; TGT for ours)."""
    t = x["t"]
    tgt = np.full(len(t), -1)
    if "TGT" in d:
        return d["TGT"].astype(int)
    pops = d["pops"]
    for i in range(len(pops) - 1, -1, -1):
        tp, b = pops[i]
        tgt[t < tp] = int(b)
    return tgt


def run_stats(kind, name):
    d = load(kind, name)
    x = derive(d)
    m = x["burn"]
    U, am, f_thr, f_sp = x["U"][m], x["am"][m], x["f_thr"][m], x["f_sp"][m]
    fm = np.linalg.norm(f_thr, axis=1)
    tgt = targets_of(d, x)
    # line of sight to the current target and the net acceleration along it
    B = d["B"]
    idx = np.flatnonzero(m)
    has = tgt[idx] >= 0
    los = np.full((len(idx), 3), np.nan)
    los[has] = B[idx[has], tgt[idx[has]], :3] - x["p"][idx[has]]
    losn = los / np.linalg.norm(los, axis=1, keepdims=True)
    anet = f_sp + G  # net accel (incl. drag)
    a_los = np.einsum("ij,ij->i", anet, losn)
    relv = np.full((len(idx), 3), np.nan)
    relv[has] = B[idx[has], tgt[idx[has]], 3:] - x["v"][idx[has]]
    closing = -np.einsum("ij,ij->i", relv, losn)  # closing speed toward the target
    s = dict(kind=kind, name=name, n=int(m.sum()), burn_s=float(m.sum() * DT),
             U_mean=float(U.mean()), U_med=float(np.median(U)),
             frac_U95=float((U > 0.95).mean()), frac_U90=float((U > 0.90).mean()), frac_U80=float((U < 0.80).mean()),
             frac_U60=float((U < 0.60).mean()),
             dv_avail=float(am.sum() * DT), dv_used=float(fm.sum() * DT), dv_lost=float(np.maximum(am - fm, 0).sum() * DT),
             fz_share=float(f_thr[:, 2].sum() / fm.sum()), fz_mean=float(f_thr[:, 2].mean()), fh_mean=float(np.linalg.norm(f_thr[:, :2], axis=1).mean()),
             anet_mean=float(np.linalg.norm(anet, axis=1).mean()), vz_net=float(anet[:, 2].sum() * DT),
             a_los_mean=float(np.nanmean(a_los)), closing_mean=float(np.nanmean(closing)),
             speed_mean=float(np.linalg.norm(x["v"][m], axis=1).mean()), alt_mean=float((x["p"][m, 2] - 20).mean()),
             frac_no_tgt=float((~has).mean()))
    if "A" in d:  # our commanded throttle, split of the lost impulse
        A = d["A"][m]
        thr = A[:, 3]
        gx, gy = np.radians(A[:, 0]), np.radians(A[:, 1])
        cosf = np.sqrt(np.clip(1 - np.sin(gx) ** 2 - np.sin(gy) ** 2, 0, 1))
        F = d["F"][m]
        a_est = d["a_thrust"][m]
        Fm = np.linalg.norm(F, axis=1)
        thr_cmd = np.where(np.isfinite(Fm), np.clip(Fm / a_est, 0, 1), 1.0)
        st = d["STRETCH"][m]
        tg = d["TGT"][m]
        s["thr_mean"] = float(thr.mean())
        s["frac_thr_full"] = float((thr > 0.99).mean())
        s["cos_mean"] = float(cosf.mean())
        s["U_model_mean"] = float((thr * cosf).mean())  # throttle x gimbal cosine = modelled utilization
        s["dv_lost_throttle"] = float(((1 - thr) * am).sum() * DT)
        s["dv_lost_gimbal"] = float((thr * (1 - cosf) * am).sum() * DT)
        s["dv_lost_ratelimit"] = float((np.maximum(thr_cmd - thr, 0) * am).sum() * DT)
        s["frac_ratelimited"] = float((np.abs(thr_cmd - thr) > 0.005).mean())
        # per speed level
        lvl = {}
        for L in STRETCH_LEVELS:
            k = (np.abs(st - L) < 1e-6) & (tg >= 0)
            lvl[str(L)] = dict(frac=float(k.mean()), thr_mean=float(thr[k].mean()) if k.any() else None,
                               dv_lost=float(((1 - thr[k]) * am[k]).sum() * DT))
        s["levels"] = lvl
        k = tg < 0
        s["no_tgt"] = dict(frac=float(k.mean()), thr_mean=float(thr[k].mean()) if k.any() else None,
                           dv_lost=float(((1 - thr[k]) * am[k]).sum() * DT))
        # _saturate "never thrust downward" clamp active: f_z = s*(1-9.81)+9.81 -> f_z near 1 when s == 1
        clamp = np.isfinite(Fm) & (F[:, 2] < 1.05) & (Fm < 0.98 * a_est)
        s["frac_clamp"] = float(clamp.mean())
        s["dv_lost_clamp"] = float(((1 - thr[clamp]) * am[clamp]).sum() * DT)
        # where the throttle was down: required force below full
        s["frac_thr_below90"] = float((thr < 0.9).mean())
    return s, d, x


def disp_profile(acc):
    """int_0^T (T - s) acc(s) ds for T = k DT (k = 0..n-1), acc sampled at DT (trapezoid-free cumulative form)."""
    c1 = np.concatenate([[0.0], np.cumsum(acc) * DT])  # int_0^T acc
    c2 = np.concatenate([[0.0], np.cumsum(c1[:-1] + 0.5 * acc * DT) * DT])  # int_0^T int_0^s acc
    return c2


def hop_stats(kind, name, d, x, r_hit=None):
    t, p, v = x["t"], x["p"], x["v"]
    L = x["L"]
    B, S = d["B"], d["S"]
    pops = d["pops"]
    i_launch = int(np.searchsorted(t, L - 1e-9))
    events = [(L, -1, i_launch)] + [(float(tp), int(b), int(np.argmax(S[:, int(b)] == 2))) for tp, b in pops]
    hops = []
    Fsp = np.linalg.norm(x["f_sp"], axis=1)
    for (t0, _, i0), (t1, b, i1) in zip(events[:-1], events[1:]):
        if t0 - L >= BURN - 0.05:
            break
        i0 = max(i0, i_launch + 1)
        n = i1 - i0
        if n <= 1:
            continue
        # hit tolerance: the distance at the detected pop step (so T_act satisfies the condition by construction)
        rh = float(np.linalg.norm(p[i1] - B[i1, b, :3])) if r_hit is None else r_hit
        T = np.arange(n + 1) * DT
        bt = B[i0:i1 + 1, b, :3].astype(float)
        miss = bt - p[i0] - v[i0] * T[:, None] - 0.5 * G * T[:, None] ** 2
        dm = np.linalg.norm(miss, axis=1)
        tau = x["tau"][i0:i1 + 1]
        burning = tau < BURN
        am = np.where(burning, x["am"][i0:i1 + 1], 0.0)
        fsp = np.where(np.isfinite(Fsp[i0:i1 + 1]), Fsp[i0:i1 + 1], 0.0)
        fsp = np.where(burning, fsp, 0.0)  # after burnout: only drag, not usable
        D_full = disp_profile(np.maximum(am, fsp)[:-1])[: n + 1]  # full thrust, never below what was realized
        D_mag = disp_profile(fsp[:-1])[: n + 1]
        ok_full = dm <= D_full + rh
        ok_mag = dm <= D_mag + rh
        T_lb = T[np.argmax(ok_full)] if ok_full.any() else np.nan
        T_mag = T[np.argmax(ok_mag)] if ok_mag.any() else np.nan
        seg = slice(i0, i1)
        bm = x["burn"][seg]
        U_hop = float(np.nanmean(x["U"][seg][bm])) if bm.any() else np.nan
        relv = B[i1 - 1, b, 3:] - v[i1 - 1]
        hops.append(dict(kind=kind, name=name, t0=t0, tau0=t0 - L, T_act=float(n * DT), T_lb=float(T_lb), T_mag=float(T_mag),
                         U_hop=U_hop, burn_frac=float(bm.mean()), rel_speed_hit=float(np.linalg.norm(relv)),
                         speed0=float(np.linalg.norm(v[i0])), r_hit=rh, first=bool(len(hops) == 0)))
    return hops


def main():
    runs, hops = [], []
    for kind in KINDS:
        for name in NAMES:
            s, d, x = run_stats(kind, name)
            runs.append(s)
            hops += hop_stats(kind, name, d, x)
    json.dump(dict(runs=runs, hops=hops), open(OUT / "util_results.json", "w"), indent=1)

    def agg(kind, key, w="burn_s"):
        rs = [r for r in runs if r["kind"] == kind]
        ww = np.array([r[w] for r in rs])
        return float(np.sum([r[key] * wi for r, wi in zip(rs, ww)]) / ww.sum())

    def tot(kind, key):
        return float(np.sum([r[key] for r in runs if r["kind"] == kind]))

    lines = ["== burn-time utilization (time-weighted over 11 runs) =="]
    keys = ["U_mean", "U_med", "frac_U95", "frac_U90", "frac_U80", "frac_U60", "fz_share", "fz_mean", "fh_mean", "anet_mean",
            "a_los_mean", "closing_mean", "speed_mean", "alt_mean", "frac_no_tgt"]
    lines.append(f"{'metric':16s}" + "".join(f"{k:>14s}" for k in KINDS))
    for k in keys:
        lines.append(f"{k:16s}" + "".join(f"{agg(kind, k):14.3f}" for kind in KINDS))
    for k in ["burn_s", "dv_avail", "dv_used", "dv_lost", "vz_net"]:
        lines.append(f"{k + ' /run':16s}" + "".join(f"{tot(kind, k) / 11:14.2f}" for kind in KINDS))
    lines.append("")
    lines.append("== ours: lost impulse split (m/s per run, burn) ==")
    for kind in KINDS[1:]:
        rs = [r for r in runs if r["kind"] == kind]
        f = lambda k: np.mean([r[k] for r in rs])  # noqa: E731
        lines.append(f"{kind}: thr_mean {f('thr_mean'):.3f} frac_thr_full {f('frac_thr_full'):.3f} frac_thr<0.9 {f('frac_thr_below90'):.3f} "
                     f"cos_mean {f('cos_mean'):.4f} U_model {f('U_model_mean'):.3f} | lost: throttle {f('dv_lost_throttle'):.1f} "
                     f"gimbal-cos {f('dv_lost_gimbal'):.1f} rate-limit {f('dv_lost_ratelimit'):.2f} (frac steps {f('frac_ratelimited'):.3f}) "
                     f"clamp {f('dv_lost_clamp'):.2f} (frac {f('frac_clamp'):.4f})")
        for L in STRETCH_LEVELS:
            fr = np.mean([r["levels"][str(L)]["frac"] for r in rs])
            th = [r["levels"][str(L)]["thr_mean"] for r in rs if r["levels"][str(L)]["thr_mean"] is not None]
            dl = np.mean([r["levels"][str(L)]["dv_lost"] for r in rs])
            lines.append(f"   stretch {L}: frac of burn {fr:.3f}  thr_mean {np.mean(th) if th else float('nan'):.3f}  dv_lost {dl:.1f} m/s/run")
        fr = np.mean([r["no_tgt"]["frac"] for r in rs]); dl = np.mean([r["no_tgt"]["dv_lost"] for r in rs])
        lines.append(f"   no target : frac of burn {fr:.3f}  dv_lost {dl:.1f} m/s/run")
    lines.append("")
    lines.append("== per-hop bounds (hops starting in the burn; s) ==")
    lines.append(f"{'kind':12s} {'n':>4s} {'T_act':>7s} {'T_lb':>7s} {'T_mag':>7s} {'mag_loss':>9s} {'dir_loss':>9s} {'U_hop':>6s} {'v_rel_hit':>9s}  (median / mean)")
    for kind in KINDS:
        for sel, lab in ((lambda h: True, "all"), (lambda h: not h["first"], "no-first"), (lambda h: h["first"], "first")):
            hs = [h for h in hops if h["kind"] == kind and sel(h) and np.isfinite(h["T_mag"]) and h["burn_frac"] > 0.5]
            a = lambda k: np.array([h[k] for h in hs])  # noqa: E731
            ml, dl = a("T_mag") - a("T_lb"), a("T_act") - a("T_mag")
            lines.append(f"{kind[:12]:12s} {len(hs):4d} {np.median(a('T_act')):7.2f} {np.median(a('T_lb')):7.2f} {np.median(a('T_mag')):7.2f} "
                         f"{np.median(ml):9.2f} {np.median(dl):9.2f} {np.median(a('U_hop')):6.3f} {np.median(a('rel_speed_hit')):9.2f}  med {lab}")
            lines.append(f"{'':12s} {'':4s} {a('T_act').mean():7.2f} {a('T_lb').mean():7.2f} {a('T_mag').mean():7.2f} "
                         f"{ml.mean():9.2f} {dl.mean():9.2f} {np.nanmean(a('U_hop')):6.3f} {a('rel_speed_hit').mean():9.2f}  mean {lab}")
    txt = "\n".join(lines)
    print(txt)
    (OUT / "util_summary.txt").write_text(txt + "\n")


if __name__ == "__main__":
    main()

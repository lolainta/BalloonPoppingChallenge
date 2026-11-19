"""Attitude / pointing diagnosis: their (leaderboard replays) vs ours_strict (same order, same launch) vs ours_free.

Outputs (stdout + analyze_out.json):
 1. body-rate |w_xy| distribution in the burn
 2. gimbal use: inferred from the angular acceleration (validated on ours: median error 0.24 deg)
 3. throttle (thrust accel / full-thrust accel)
 4. ours: angle between the nose (= thrust axis) and the wanted specific force F
 5. per hop: hindsight direction f_h(t) = the constant specific force that reaches the popped balloon exactly at the pop
    time from the current state; required turn at hop start, time until the nose is within 20 / 10 deg of f_h,
    peak rate while re-pointing, fraction of the hop mis-pointed, lost thrust
 6. ours: re-point delay after each pop / target change measured against the GNC's own wanted F
"""
import json
import numpy as np
from common import *

KINDS = ("their", "ours_strict", "ours_free")
EDGE = 0.5  # (s) skip the first 0.5 s after launch (rail / initial pitch-over)


def a_full_table():
    """full-throttle thrust accel vs time since launch, from our logs (throttle > 0.98, small gimbal)."""
    tau_all, a_all = [], []
    for n in NAMES:
        d = load("ours_strict", n)
        b = d["burn"] & (d["A"][:, 3] > 0.98) & (np.hypot(d["A"][:, 0], d["A"][:, 1]) < 5)
        b[:d["i0"] + 50] = False
        tau_all.append(d["t"][b] - d["t"][d["i0"]]); a_all.append(d["Tm"][b])
    tau, a = np.concatenate(tau_all), np.concatenate(a_all)
    grid = np.arange(0, 30.5, 1.0)
    med = np.array([np.median(a[(tau >= g - 0.5) & (tau < g + 0.5)]) if ((tau >= g - 0.5) & (tau < g + 0.5)).sum() > 20 else np.nan for g in grid])
    ok = np.isfinite(med)
    return lambda x: np.interp(x, grid[ok], med[ok])


def hop_metrics(d, a_full):
    t, p, v, zb, B = d["t"], d["p"], d["v"], d["zb"], d["B"]
    res = []
    for ts, te, b, first in hops(d):
        i_s, i_e = idx(t, ts), idx(t, te)
        i_pb = max(i_e - 1, 0)
        pb = B[i_pb, b, :3].astype(float)  # balloon position just before the pop
        ii = np.arange(i_s + 1, i_e)
        tau = te - t[ii]
        ok = tau >= 0.3
        ii, tau = ii[ok], tau[ok]
        if len(ii) < 5:
            continue
        fh = 2 * (pb - p[ii] - v[ii] * tau[:, None]) / tau[:, None] ** 2 - G
        th = np.degrees(ang(zb[ii], fh))
        tt = t[ii] - ts
        def first_below(x):
            k = np.flatnonzero(th < x)
            return float(tt[k[0]]) if len(k) else np.nan
        af = a_full(t[ii] - t[d["i0"]])
        w = d["wxy"][ii]
        t10 = first_below(10.0)
        jj = ii[: max(1, np.searchsorted(tt, t10 if np.isfinite(t10) else tt[-1]) + 1)]
        res.append(dict(
            hop=te - ts, first=bool(first), th0=float(th[0]), t20=first_below(20.0), t10=t10,
            wpk=float(d["wxy"][jj].max()), frac20=float((th > 20).mean()), frac10=float((th > 10).mean()),
            fh0=float(np.linalg.norm(fh[0]) / af[0]), fh_med=float(np.median(np.linalg.norm(fh, axis=1) / af)),
            lost=float(np.sum((1 - np.cos(np.radians(th))) * d["Tm"][ii]) * DT),  # thrust-accel*s not along f_h
            lost_perp=float(np.sum(np.sin(np.radians(th)) * d["Tm"][ii]) * DT),
            thr=float(np.median(d["Tm"][ii] / af)),
        ))
    return res


def main():
    a_full = a_full_table()
    out = {}
    print("=== 1-3: body rate, gimbal, throttle in the burn (pooled over 11 runs; first 0.5 s skipped) ===")
    for k in KINDS:
        W, GI, GX, TH, SATX = [], [], [], [], []
        per = []
        for n in NAMES:
            d = load(k, n)
            b = d["burn"].copy(); b[:d["i0"] + int(EDGE / DT)] = False; b[-2:] = False
            W.append(d["wxy"][b]); GI.append(d["gimb"][b]); GX.append(np.abs(d["gxy"][b]))
            TH.append(d["Tm"][b] / a_full(d["t"][b] - d["t"][d["i0"]]))
            per.append((n, np.percentile(d["wxy"][b], 95), d["wxy"][b].max()))
            if k != "their":
                SATX.append(np.hypot(d["A"][b, 0], d["A"][b, 1]))
        W, GI, GX, TH = map(np.concatenate, (W, GI, GX, TH))
        r = dict(
            w=pct(W) | dict(max=float(W.max()), frac_gt1=float((W > 1).mean()), frac_gt1_5=float((W > 1.5).mean()),
                            frac_gt2=float((W > 2).mean()), frac_gt2_5=float((W > 2.5).mean())),
            gimbal_inferred=pct(GI) | dict(frac_ge14=float((GI >= 14).mean()), frac_ge10=float((GI >= 10).mean()),
                                            frac_gt16_box=float((GI > 16).mean()),
                                            frac_axis_ge14=float((GX.max(1) >= 14).mean())),
            throttle=pct(TH, (10, 50, 90)) | dict(mean=float(TH.mean()), frac_ge0_97=float((TH >= 0.97).mean()),
                                                  frac_le0_7=float((TH <= 0.7).mean())),
            per_run_w=[(n, round(float(a), 2), round(float(m), 2)) for n, a, m in per],
        )
        if SATX:
            S = np.concatenate(SATX); r["gimbal_logged"] = dict(frac_ge14_5=float((S >= 14.5).mean()), p50=float(np.median(S)), p90=float(np.percentile(S, 90)))
        out[k] = r
        print(f"\n[{k}]")
        print("  |w_xy| rad/s  " + "  ".join(f"{a}={b:.2f}" for a, b in r["w"].items()))
        print("  gimbal (inferred) " + "  ".join(f"{a}={b:.3f}" for a, b in r["gimbal_inferred"].items()))
        if SATX:
            print("  gimbal (logged)   " + "  ".join(f"{a}={b:.3f}" for a, b in r["gimbal_logged"].items()))
        print("  throttle          " + "  ".join(f"{a}={b:.3f}" for a, b in r["throttle"].items()))
        print("  per-run p95/max |w|: " + ", ".join(f"{n} {a}/{m}" for n, a, m in r["per_run_w"]))

    print("\n=== 4: ours, angle between nose (thrust axis) and wanted F, guided burn steps ===")
    for k in ("ours_strict", "ours_free"):
        TH, WT, EV = [], [], []
        for n in NAMES:
            d = load(k, n)
            b = d["burn"] & np.isfinite(d["F"]).all(1); b[:d["i0"] + int(EDGE / DT)] = False
            th = np.degrees(ang(d["zb"], np.nan_to_num(d["F"])))
            TH.append(th[b]); WT.append(np.linalg.norm(d["F"][b], axis=1))
            # 6: events = pops and target changes; delay until the nose is within 20 / 10 deg of F
            tgt = d["TGT"]; t = d["t"]
            ev = np.flatnonzero((tgt[1:] != tgt[:-1]) & (tgt[1:] >= 0) & b[1:]) + 1
            for e in ev:
                jj = np.arange(e, min(e + 500, len(t)))
                jj = jj[b[jj] & (tgt[jj] == tgt[e])]
                if len(jj) < 3:
                    continue
                x = th[jj]; tt = t[jj] - t[e]
                f20 = np.flatnonzero(x < 20); f10 = np.flatnonzero(x < 10)
                EV.append((x[0], tt[f20[0]] if len(f20) else np.nan, tt[f10[0]] if len(f10) else np.nan,
                           d["wxy"][jj[: (f10[0] + 1) if len(f10) else len(jj)]].max()))
        TH, WT = np.concatenate(TH), np.concatenate(WT)
        EV = np.array(EV)
        big = EV[:, 0] > 30
        r = dict(angle=pct(TH) | dict(mean=float(TH.mean()), frac_gt10=float((TH > 10).mean()), frac_gt20=float((TH > 20).mean()),
                                        frac_gt45=float((TH > 45).mean()),
                                        thrust_eff=float(np.sum(WT * np.cos(np.radians(TH))) / np.sum(WT))),
                 events=dict(n=int(len(EV)), n_big=int(big.sum()), th0_med=float(np.median(EV[:, 0])),
                             t20_med_big=float(np.nanmedian(EV[big, 1])), t10_med_big=float(np.nanmedian(EV[big, 2])),
                             t10_p90_big=float(np.nanpercentile(EV[big, 2], 90)),
                             rate_deg_s_big=float(np.nanmedian((EV[big, 0] - 10) / EV[big, 2])),
                             wpk_med_big=float(np.nanmedian(EV[big, 3]))))
        out[k]["pointing_vs_F"] = r
        print(f"[{k}] angle(nose, F) deg " + "  ".join(f"{a}={b:.3f}" for a, b in r["angle"].items()))
        print(f"[{k}] events (target changes incl. after pops): " + "  ".join(f"{a}={b:.3f}" for a, b in r["events"].items()))

    print("\n=== 5: per hop vs hindsight direction f_h (constant specific force that pops the balloon at the actual pop time) ===")
    for k in KINDS:
        H = []
        for n in NAMES:
            H += hop_metrics(load(k, n), a_full)
        out[k]["hops"] = H
        sub = [h for h in H if not h["first"]]
        big = [h for h in sub if h["th0"] > 30]
        med = lambda key, L: float(np.nanmedian([h[key] for h in L])) if L else np.nan
        s = dict(n=len(sub), hop_med=med("hop", sub), th0_med=med("th0", sub), t20_med=med("t20", sub), t10_med=med("t10", sub),
                 n_th0_gt30=len(big), t10_med_big=med("t10", big), wpk_med_big=med("wpk", big),
                 rate_deg_s_big=float(np.nanmedian([(h["th0"] - 10) / h["t10"] for h in big])) if big else np.nan,
                 frac20_med=med("frac20", sub), lost_med=med("lost", sub), lost_perp_med=med("lost_perp", sub),
                 fh0_med=med("fh0", sub), fh_med=med("fh_med", sub), thr_med=med("thr", sub),
                 t10_over_hop=float(np.nanmedian([h["t10"] / h["hop"] for h in sub])))
        out[k]["hop_summary"] = s
        print(f"[{k}] " + "  ".join(f"{a}={b:.3f}" if isinstance(b, float) else f"{a}={b}" for a, b in s.items()))
    json.dump(out, open(OUT / "analyze_out.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()

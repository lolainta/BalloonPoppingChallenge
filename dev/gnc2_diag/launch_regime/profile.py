"""Launch attitude, first-seconds profile, first-pop geometry, burn-regime statistics for their / ours_strict / ours_free.

.venv/bin/python dev/gnc2_diag/launch_regime/profile.py > dev/gnc2_diag/launch_regime/profile.txt
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BURN, KINDS, NAMES, a_max, incl_head, load  # noqa: E402

OUT = Path(__file__).resolve().parent
TS = [0.5, 1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 25, 30]


def at(d, tl, key):
    i = d["i0"] + int(round(tl / 0.01))
    return d[key][min(i, len(d["t"]) - 2)]


rows = {k: [] for k in KINDS}
prof = {k: {} for k in KINDS}
for k in KINDS:
    for n in NAMES:
        d = load(k, n)
        i0, R, tl = d["i0"], d["R"], d["tl"]
        inc0, hd0 = incl_head(d["zb"][i0])
        pops = d["pops"]
        r = dict(name=n, launch=d["t_launch"], incl0=float(inc0), head0=float(hd0), n_pops=len(pops))
        # the first balloon: where it was at launch / at pop
        if len(pops):
            tp, b = pops[0]; b = int(b)
            ip = int(np.argmin(np.abs(d["t"] - tp)))
            bp = d["B"][ip, b, :3]
            bl = d["B"][i0, b, :3]
            r.update(first_dt=tp - d["t_launch"], first_b=b, first_alt=float(bp[2] - 20), first_hdist=float(np.hypot(*bp[:2])),
                     first_dist_at_launch=float(np.linalg.norm(bl - R[i0, :3])), first_b_alt_at_launch=float(bl[2] - 20),
                     first_b_airborne_at_launch=int(d["S"][i0, b] == 1),
                     first_rel_speed=float(np.linalg.norm(R[ip, 3:6] - d["B"][ip, b, 3:6])),
                     first_speed=float(np.linalg.norm(R[ip, 3:6])))
            # direction from pad to the first balloon (at pop) vs launch heading
            _, hb = incl_head(np.r_[bp[:2] - R[i0, :2], 0.0])
            r["first_bearing"] = float(hb)
            r["head_err"] = float((hd0 - hb + 180) % 360 - 180)
            ib, _ = incl_head(bp - R[i0, :3])
            r["first_elev_angle"] = float(ib)
        # burn window
        w = (tl >= 0) & (tl <= BURN) & np.isfinite(d["f"][:, 0])
        f, v, alt = d["f"][w], R[w, 3:6], d["alt"][w]
        amax = a_max(tl[w])
        sp = np.linalg.norm(v, axis=1)
        e = 0.5 * sp**2 + 9.81 * alt
        r.update(
            alt_mean=float(alt.mean()), alt_med=float(np.median(alt)), alt_max=float(alt.max()),
            frac_alt_lt5=float((alt < 5).mean()), frac_alt_lt10=float((alt < 10).mean()), frac_alt_lt20=float((alt < 20).mean()),
            speed_mean=float(sp.mean()), vz_mean=float(v[:, 2].mean()), vh_mean=float(np.hypot(v[:, 0], v[:, 1]).mean()),
            thr_frac=float((np.linalg.norm(f, axis=1) / amax).mean()),  # mean |f| / full-thrust accel
            fz_mean=float(f[:, 2].mean()), fh_mean=float(np.hypot(f[:, 0], f[:, 1]).mean()),
            frac_fz_lt_g=float((f[:, 2] < 9.81).mean()), frac_fz_lt1=float((f[:, 2] < 1.0).mean()), frac_fz_lt0=float((f[:, 2] < 0).mean()),
            power_mean=float((f * v).sum(1).mean()),  # thrust work per kg per s (m^2/s^3)
            e_end=float(e[-1]), grav_loss=float((9.81 * v[:, 2] / np.maximum(sp, 1e-3)).sum() * 0.01),
            pops_burn=int(((pops[:, 0] - d["t_launch"]) <= BURN).sum()) if len(pops) else 0,
            pops_after=int(((pops[:, 0] - d["t_launch"]) > BURN).sum()) if len(pops) else 0,
            last_pop_dt=float(pops[-1, 0] - d["t_launch"]) if len(pops) else np.nan,
        )
        # rocket altitude at pops, pop altitude of balloons
        if len(pops):
            ips = [int(np.argmin(np.abs(d["t"] - tp))) for tp in pops[:, 0]]
            r["pop_alt_med"] = float(np.median([d["alt"][i] for i in ips]))
            r["pop_alt_list"] = [round(float(d["alt"][i]), 1) for i in ips]
            r["pop_dt_list"] = [round(float(tp - d["t_launch"]), 2) for tp in pops[:, 0]]
        # ours: ground-avoidance rule activity (from logged F, post-saturate): F_z == 1.0 means the "never thrust downward"
        # clamp bound (a_cmd_z clipped at 1 - g with s = 1); min_alt rule: alt < 3 and vz < 0 while guiding
        if "F" in d:
            F = d["F"][w]; g = np.isfinite(F[:, 0])
            r["frac_Fz_clamp"] = float((np.abs(F[g, 2] - 1.0) < 1e-3).mean())
            r["frac_Fz_lt2"] = float((F[g, 2] < 2.0).mean())
            r["frac_minalt_rule"] = float(((alt < 3) & (v[:, 2] < 0))[g].mean())
            r["thr_cmd_mean"] = float(d["A"][w, 3].mean())
            r["frac_thr_lt095"] = float((d["A"][w, 3] < 0.95).mean())
            r["stretch_mean"] = float(np.nanmean(d["STRETCH"][w]))
        rows[k].append(r)
        # profile at fixed times since launch
        for T in TS:
            z = at(d, T, "zb"); inc, _ = incl_head(z)
            vv = at(d, T, "R")[3:6]; a = float(at(d, T, "alt"))
            prof[k].setdefault(T, []).append([a, np.linalg.norm(vv), vv[2], np.hypot(vv[0], vv[1]), float(inc),
                                              0.5 * vv @ vv + 9.81 * a, np.hypot(*(at(d, T, "R")[:2]))])

json.dump(rows, open(OUT / "profile_rows.json", "w"), indent=1, default=float)


def med(k, key):
    x = np.array([r.get(key, np.nan) for r in rows[k]], float)
    return np.nanmedian(x), np.nanmean(x)


print("=== per-run launch attitude and first pop ===")
for i, n in enumerate(NAMES):
    s = []
    for k in KINDS:
        r = rows[k][i]
        s.append(f"{r['incl0']:5.1f}/{r['head0']:5.0f} dt1 {r.get('first_dt', np.nan):5.2f} b{r.get('first_b', -1):3d} "
                 f"alt {r.get('first_alt', np.nan):5.1f} d0 {r.get('first_dist_at_launch', np.nan):5.1f} herr {r.get('head_err', np.nan):5.0f}")
    print(f"{n:15s} | " + " | ".join(s))

keys = ["incl0", "first_dt", "first_alt", "first_hdist", "first_dist_at_launch", "first_b_alt_at_launch", "first_b_airborne_at_launch",
        "first_elev_angle", "first_speed", "first_rel_speed", "alt_mean", "alt_med", "alt_max", "frac_alt_lt5", "frac_alt_lt10",
        "frac_alt_lt20", "speed_mean", "vz_mean", "vh_mean", "thr_frac", "fz_mean", "fh_mean", "frac_fz_lt_g", "frac_fz_lt1",
        "frac_fz_lt0", "power_mean", "e_end", "grav_loss", "pops_burn", "pops_after", "last_pop_dt", "pop_alt_med",
        "frac_Fz_clamp", "frac_Fz_lt2", "frac_minalt_rule", "thr_cmd_mean", "frac_thr_lt095", "stretch_mean"]
print("\n=== aggregate: median (mean) over 11 runs ===")
print(f"{'metric':28s}" + "".join(f"{k:>22s}" for k in KINDS))
for key in keys:
    print(f"{key:28s}" + "".join("{:>13.2f} ({:6.2f})".format(*med(k, key)) for k in KINDS))
print("\nsums: pops_burn / pops_after:", {k: (sum(r['pops_burn'] for r in rows[k]), sum(r['pops_after'] for r in rows[k])) for k in KINDS})

print("\n=== profile vs time since launch: median over runs [alt, speed, vz, vh, body incl, spec. energy, horiz dist from pad] ===")
for T in TS:
    print(f"t+{T:4.1f}s " + " | ".join(" ".join(f"{x:6.1f}" for x in np.median(np.array(prof[k][T]), 0)) for k in KINDS))

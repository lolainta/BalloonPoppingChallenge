"""Regime numbers over t+0..30 s: mean speed, altitude, specific energy at burnout, thrust power f.v, |f| vs full thrust,
time below 5 m, min_alt condition, logged-clamp fraction, stretch levels.  Drag removed with the still-air model only where noted."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from vc import KINDS, NAMES, get

ELEV = 20.0
M_GRAIN = np.pi * (0.0843**2 - 0.0295**2) * 0.2757 * 900
M0 = 70 + 13.9 + 2.13 + M_GRAIN
KD = 0.5 * 0.25 * np.pi * 0.22**2
for k in KINDS:
    A = {x: [] for x in "spd alt E fv ratio unused below5 below5late minalt clamp stretch thr hspd vz".split()}
    for n in NAMES:
        d = get(k, n); s = d["s"]; R = d["R"]
        w = (s >= 0) & (s < 30) & np.isfinite(d["fsp"][:, 0])
        v = R[:, 3:6]; alt = R[:, 2] - ELEV
        rho = 1.225 * (1 - 2.2558e-5 * R[:, 2]) ** 4.2559
        m = M0 - 0.425 * np.clip(s, 0, 30)
        drag = -KD * rho[:, None] * np.linalg.norm(v, axis=1, keepdims=True) * v / m[:, None]
        f = d["fsp"] - drag
        amax = 1080.0 / m
        A["spd"].append(np.linalg.norm(v[w], axis=1).mean()); A["alt"].append(alt[w].mean())
        A["hspd"].append(np.linalg.norm(v[w, :2], axis=1).mean()); A["vz"].append(v[w, 2].mean())
        ib = d["i0"] + 2999
        A["E"].append(0.5 * v[ib] @ v[ib] + 9.81 * alt[ib])
        A["fv"].append((f[w] * v[w]).sum(1).mean())
        fn = np.linalg.norm(f[w], axis=1)
        A["ratio"].append((fn / amax[w]).mean()); A["unused"].append((amax[w] - fn).mean())
        A["below5"].append((alt[w] < 5).mean()); A["below5late"].append((alt[w & (s > 6)] < 5).mean())
        A["minalt"].append(((alt[w] < 3) & (v[w, 2] < 0)).mean())
        if "F" in d:
            F = d["F"]; g = w & np.isfinite(F[:, 2])
            A["clamp"].append((np.abs(F[g, 2] - 1.0) < 1e-6).mean())
            A["stretch"].append(np.nanmean(d["STRETCH"][w])); A["thr"].append(d["A"][w, 3].mean())
    print(f"{k:12s} speed {np.mean(A['spd']):5.1f} (horiz {np.mean(A['hspd']):5.1f}, vz {np.mean(A['vz']):5.1f})  alt {np.mean(A['alt']):6.1f}  E@burnout {np.mean(A['E']):6.0f} J/kg"
          f"  f.v {np.mean(A['fv']):5.1f} W/kg  |f|/amax {np.mean(A['ratio']):.3f} unused {np.mean(A['unused']):.2f}"
          f"  below5m {np.mean(A['below5']):.3f} (after t+6 {np.mean(A['below5late']):.4f})  minalt-cond {np.mean(A['minalt']):.4f}"
          + (f"  logged Fz==1 {np.mean(A['clamp']):.3f}  stretch {np.mean(A['stretch']):.2f} throttle {np.mean(A['thr']):.3f}" if A["clamp"] else ""))
    print(f"{'':12s} per-run speed {np.round(A['spd'],1).tolist()}")

"""Energy regime during the burn: power decomposition (|f|, |v|, cos(f,v)), vertical usage of thrust, and per-hop
gravity cost: hop time vs rocket climb between consecutive pops (least squares), their vs ours.
.venv/bin/python dev/gnc2_diag/launch_regime/regime.py
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BURN, KINDS, NAMES, a_max, load  # noqa: E402

for k in KINDS:
    P = []; hop = []
    for n in NAMES:
        d = load(k, n)
        w = (d["tl"] >= 1) & (d["tl"] <= BURN) & np.isfinite(d["f"][:, 0])
        f, v = d["f"][w], d["R"][w, 3:6]
        fm, vm = np.linalg.norm(f, axis=1), np.linalg.norm(v, axis=1)
        c = (f * v).sum(1) / np.maximum(fm * vm, 1e-6)
        # thrust split: vertical support vs rest
        P.append([fm.mean(), vm.mean(), c.mean(), (f * v).sum(1).mean(), (f[:, 2] * v[:, 2]).mean(), np.mean(np.hypot(f[:, 0], f[:, 1])),
                  (a_max(d["tl"][w]) - fm).mean()])
        p = d["pops"]
        ip = [int(np.argmin(np.abs(d["t"] - tp))) for tp in p[:, 0]]
        for a, b in zip(ip[:-1], ip[1:]):
            if d["tl"][b] > BURN:
                continue
            dz = d["alt"][b] - d["alt"][a]
            dist = np.linalg.norm(d["R"][b, :3] - d["R"][a, :3])
            hop.append([d["t"][b] - d["t"][a], dz, dist])
    P = np.array(P); H = np.array(hop)
    X = np.c_[np.ones(len(H)), np.maximum(H[:, 1], 0), np.maximum(-H[:, 1], 0), H[:, 2]]
    coef = np.linalg.lstsq(X, H[:, 0], rcond=None)[0]
    print(f"{k:12s} |f| {P[:,0].mean():5.2f}  |v| {P[:,1].mean():5.2f}  cos(f,v) {P[:,2].mean():5.2f}  f.v {P[:,3].mean():6.1f}  fz*vz {P[:,4].mean():6.1f}  "
          f"|f_h| {P[:,5].mean():5.2f}  unused thrust {P[:,6].mean():5.2f} m/s^2")
    print(f"{'':12s} hops(burn) {len(H)}  median dt {np.median(H[:,0]):4.2f}  median climb {np.median(H[:,1]):5.1f} m  frac climbing {np.mean(H[:,1]>0):.2f}  "
          f"median straight dist {np.median(H[:,2]):5.1f} m  mean straight-line speed {np.median(H[:,2]/H[:,0]):5.2f} m/s")
    print(f"{'':12s} fit dt = {coef[0]:.2f} + {coef[1]:.3f}*climb + {coef[2]:.3f}*descent + {coef[3]:.3f}*dist   (s, m)")

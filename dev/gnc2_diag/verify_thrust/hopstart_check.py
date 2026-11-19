"""Verifier: hop-start velocity bonus (T_lb from rest-relative-to-balloon minus T_lb from actual v0), closing speed and
cos(v_rel, LOS) at the pop that starts the hop, nose at pop vs thrust-weighted nose over the next 1 s."""
import sys; sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
import numpy as np
from common import *
from util import run
from breakdown import pops_from_S
from hops import D_of

def go(kind, sims=False):
    rows, ang, turn = [], [], []
    for n in NAMES:
        r = run(kind, n, sims); d, x = r["d"], r["x"]; t, p, v, B = x["t"], x["p"], x["v"], d["B"]
        P = pops_from_S(d); am = r["am_all"]; thr = x["thr_ax"]
        for (t0, _, i0), (t1, b, i1) in zip(P[:-1], P[1:]):
            if t[i0] - x["L"] >= BURN - 0.05 or i1 - i0 < 2: continue
            H = min(800, len(t) - 1 - i0); T = np.arange(H + 1) * DT
            taum = x["taum"][i0 + 1:i0 + H + 1]
            D = D_of(np.where(taum < BURN, am[i0 + 1:i0 + H + 1], 0.0))
            bt = B[i0:i0 + H + 1, b, :3].astype(float); rh = float(np.linalg.norm(p[i1] - B[i1, b, :3]))
            def tlb(v0):
                ok = np.linalg.norm(bt - p[i0] - v0 * T[:, None] - 0.5 * G * T[:, None]**2, axis=1) <= D + rh
                return T[np.argmax(ok)] if ok.any() else np.nan
            vb = B[i0, b, 3:].astype(float); r0 = B[i0, b, :3] - p[i0]; los = r0 / np.linalg.norm(r0)
            vr = v[i0] - vb
            rows.append((tlb(v[i0]), tlb(vb), vr @ los, vr @ los / max(np.linalg.norm(vr), 1e-6), np.linalg.norm(v[i0]), (i1 - i0) * DT))
            if t[i0] - x["L"] <= BURN - 1.5 and i0 + 100 < len(t):
                w = np.clip(np.nan_to_num(thr[i0 + 1:i0 + 101]), 0, None)[:, None]
                fd = (x["nmid"][i0 + 1:i0 + 101] * w).sum(0); fd /= np.linalg.norm(fd)
                fp = (x["nmid"][i0 - 49:i0 + 1] * np.clip(np.nan_to_num(thr[i0 - 49:i0 + 1]), 0, None)[:, None]).sum(0); fp /= np.linalg.norm(fp)
                ang.append(np.degrees(np.arccos(np.clip(x["nz"][i0] @ fd, -1, 1)))); turn.append(np.degrees(np.arccos(np.clip(fp @ fd, -1, 1))))
    a = np.array(rows); ok = np.isfinite(a[:, 0]) & np.isfinite(a[:, 1])
    return (f"{kind:12s} n={ok.sum():3d} T_lb mean {a[ok,0].mean():.2f} T_rest mean {a[ok,1].mean():.2f} bonus mean {(a[ok,1]-a[ok,0]).mean():+.2f} "
            f"| closing med {np.median(a[:,2]):.1f} cos med {np.median(a[:,3]):.2f} speed med {np.median(a[:,4]):.1f} | nose-vs-next1s med {np.median(ang):.1f} "
            f"thrust turn med {np.median(turn):.1f} (n={len(ang)})")

lines = [go(k) for k in ("their", "ours_strict", "ours_free")] + [go(k, True) for k in ("s08", "fullsat", "lead10")]
print("\n".join(lines)); (ROOT / "dev/gnc2_diag/verify_thrust/hopstart_check.txt").write_text("\n".join(lines) + "\n")

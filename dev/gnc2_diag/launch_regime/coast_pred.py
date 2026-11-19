"""How predictable is the post-burnout path?  Point-mass propagation (gravity + drag, air moving with the balloons' median
horizontal velocity) from the burnout state vs the actual path, at +1/+2/+3/+5 s; vs body-velocity angle at burnout.
.venv/bin/python dev/gnc2_diag/launch_regime/coast_pred.py
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BURN, K_DRAG, KINDS, M0, NAMES, load  # noqa: E402

m_end = M0 - 0.425 * BURN
for k in KINDS:
    rows = []
    for n in NAMES:
        d = load(k, n)
        ib = d["i0"] + int(round(BURN / 0.01)) + 1
        R = d["R"]
        if ib >= len(R) or not np.isfinite(R[ib, 0]):
            continue
        fly = d["S"][ib] == 1
        w = np.r_[np.median(d["B"][ib, fly, 3:5], 0), 0.0]
        p, v = R[ib, :3].astype(float), R[ib, 3:6].astype(float)
        ang = np.degrees(np.arccos(np.clip(d["zb"][ib] @ v / np.linalg.norm(v), -1, 1)))
        err = []
        for T in (1, 2, 3, 5):
            pp, vv = p.copy(), v.copy()
            for _ in range(int(T / 0.01)):
                vr = vv - w
                rho = 1.225 * np.clip(1.0 - 2.2558e-5 * pp[2], 0.5, 1.0) ** 4.2559
                a = np.array([0, 0, -9.81]) - K_DRAG * rho * np.linalg.norm(vr) * vr / m_end
                vv = vv + a * 0.01; pp = pp + vv * 0.01
            j = ib + int(T / 0.01)
            err.append(np.linalg.norm(pp - R[j, :3]) if j < len(R) and np.isfinite(R[j, 0]) and d["alt"][j] > 0 else np.nan)
        rows.append([ang] + err)
    A = np.array(rows)
    print(f"{k:12s} body-vel angle med {np.nanmedian(A[:,0]):5.1f}   point-mass error median at +1/+2/+3/+5 s: " + " / ".join(f"{np.nanmedian(A[:,i]):5.1f}" for i in range(1, 5)) + " m")
    lo, hi = A[:, 0] < 45, A[:, 0] >= 45
    print(f"{'':12s} angle<45 (n={lo.sum()}): " + " / ".join(f"{np.nanmedian(A[lo,i]):5.1f}" for i in range(1, 5)) +
          f"    angle>=45 (n={hi.sum()}): " + " / ".join(f"{np.nanmedian(A[hi,i]):5.1f}" for i in range(1, 5)))

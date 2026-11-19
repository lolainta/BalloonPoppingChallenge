"""Hop-start state: range to the next target, start speed, and how much the start velocity helps the full-thrust bound:
T_lb(actual v0) vs T_lb(v0 = balloon velocity, i.e. starting at rest relative to the target). Non-first hops in the burn."""
import sys; sys.path.insert(0, 'dev/gnc2_diag/thrust_util')
import numpy as np
from util_analysis import load, derive, NAMES, DT, G, BURN, disp_profile
res = {}
import sys as _s
from pathlib import Path as _P
SIMS = _P("dev/gnc2_diag/thrust_util/sims")
def _load(kind, n):
    if kind in ("their", "ours_strict", "ours_free"):
        return load(kind, n)
    return {k: v for k, v in np.load(SIMS / f"{kind}_{n}.npz").items()}
for kind in (_s.argv[1:] or ["their", "ours_strict", "ours_free"]):
    rows = []
    for n in NAMES:
        d = _load(kind, n); x = derive(d); t, p, v = x["t"], x["p"], x["v"]; B, S = d["B"], d["S"]
        ev = [(float(tp), int(b), int(np.argmax(S[:, int(b)] == 2))) for tp, b in d["pops"]]
        for (t0, _, i0), (t1, b, i1) in zip(ev[:-1], ev[1:]):
            if t0 - x["L"] >= BURN - 0.05 or i1 - i0 < 2:
                continue
            H = 800; T = np.arange(H + 1) * DT
            # extrapolate the balloon past the log end with its last velocity (only needed for the v0 = vb case)
            j = np.minimum(i0 + np.arange(H + 1), len(t) - 1)
            bt = B[j, b, :3].astype(float) + (i0 + np.arange(H + 1) - j)[:, None] * DT * B[j, b, 3:].astype(float)
            tau = x["tau"][i0:i0 + H + 1]; tau = np.concatenate([tau, tau[-1] + DT * np.arange(1, H + 2 - len(tau))])
            am = np.where(tau < BURN, x["am"][0] * 0 + (1.0097 * 1080 / (90.891 - 0.587 * np.clip(tau, 0, 30))), 0.0)
            D = disp_profile(am[:-1])
            rh = float(np.linalg.norm(p[i1] - B[i1, b, :3]))
            def tlb(v0):
                miss = bt - p[i0] - v0 * T[:, None] - 0.5 * G * T[:, None] ** 2
                ok = np.linalg.norm(miss, axis=1) <= D + rh
                return T[np.argmax(ok)] if ok.any() else np.nan
            r0 = B[i0, b, :3] - p[i0]
            vb = B[i0, b, 3:].astype(float)
            rows.append(dict(T_act=(i1 - i0) * DT, T_lb=tlb(v[i0]), T_rest=tlb(vb), rng=np.linalg.norm(r0), dz=r0[2],
                             speed0=np.linalg.norm(v[i0]), vclose0=float((v[i0] - vb) @ r0 / np.linalg.norm(r0)),
                             cosang=float((v[i0] - vb) @ r0 / np.linalg.norm(r0) / max(np.linalg.norm(v[i0] - vb), 1e-6))))
    a = lambda k: np.array([r[k] for r in rows], float)  # noqa: E731
    ok = np.isfinite(a("T_lb")) & np.isfinite(a("T_rest"))
    print(f"{kind:12s} n={ok.sum():3d} T_act med {np.median(a('T_act')[ok]):.2f} | T_lb med {np.median(a('T_lb')[ok]):.2f} mean {a('T_lb')[ok].mean():.2f} | "
          f"T_lb(v0=vb) med {np.median(a('T_rest')[ok]):.2f} mean {a('T_rest')[ok].mean():.2f} | velocity bonus mean {(a('T_rest') - a('T_lb'))[ok].mean():+.2f} | "
          f"range med {np.median(a('rng')):.1f} dz med {np.median(a('dz')):+.1f} speed0 med {np.median(a('speed0')):.1f} "
          f"closing0 med {np.median(a('vclose0')):+.1f} cos(vrel, LOS) med {np.median(a('cosang')):+.2f}")

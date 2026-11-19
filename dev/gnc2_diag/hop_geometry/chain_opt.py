"""Offline estimate of the chaining gain (constant-force point-mass model, gravity, balloons at constant velocity).
From the state at pop k, two legs: hit b_{k+1} at time t_h (aim point b + delta, |delta| <= DMAX, perpendicular to the
relative velocity), then min time to b_{k+2} from the arrival state.
  base : t_h = STRETCH * Tmin_centre, delta = 0          (what N=2 ZEM guidance does, ignoring the next balloon)
  opt  : min over t_h in [Tmin, 2.5 Tmin] and delta       (terminal shaping toward the next balloon)
  opt0 : same with delta = 0 (hit-time shaping only)
Two-leg time saved = base - opt. Run on the hop starts of their / ours_strict / ours_free (their order of balloons)."""
import sys
import numpy as np, pandas as pd
import hops as H
DMAX = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
G = H.G


def leg(p, v, bp, bv, t, delta):
    miss = bp - p + (bv - v) * t - 0.5 * G * t * t + delta
    f = 2 * miss / (t * t)
    return f, p + v * t + 0.5 * (f + G) * t * t, v + (f + G) * t


def two_leg(p, v, b1p, b1v, b2p, b2v, a, t_h, delta):
    f, ph, vh = leg(p, v, b1p, b1v, t_h, delta)
    if np.linalg.norm(f) > a:
        return np.inf
    t2, _ = H.tmin(ph, vh, b2p + b2v * t_h, b2v, a)
    return t_h + t2


def run(kind, stretch=1.0):
    rows = []
    for n in H.NAMES:
        z = np.load(H.D / f"{kind}_{n}.npz"); t, R, B = z["t"], z["R"], z["B"]; launch = float(z["launch"])
        live = ~np.isnan(R[:, 0]); idx = np.flatnonzero(live)
        fa = np.full_like(R[:, :3], np.nan); fa[idx] = H.thrust_acc(t[idx], R[idx]); fm = np.linalg.norm(fa, axis=1)
        a = float(np.nanpercentile(fm[live & (t > launch + .5) & (t < launch + 29.5)], 95))
        P = [(float(tp), int(b)) for tp, b in z["pops"] if tp <= launch + 30]
        for k in range(len(P) - 2):
            (t0, _), (t1, b1), (t2_, b2) = P[k], P[k + 1], P[k + 2]
            j = int(np.argmin(np.abs(t - t0))) - 1
            p, v = R[j, :3], R[j, 3:6]
            b1p, b1v, b2p, b2v = [B[j, b, s].astype(float) for b in (b1, b2) for s in (slice(0, 3), slice(3, 6))][0:4]
            b1p, b1v = B[j, b1, :3].astype(float), B[j, b1, 3:].astype(float)
            b2p, b2v = B[j, b2, :3].astype(float), B[j, b2, 3:].astype(float)
            tc, _ = H.tmin(p, v, b1p, b1v, a, 0.0)
            if not np.isfinite(tc):
                continue
            base = two_leg(p, v, b1p, b1v, b2p, b2v, a, max(stretch * tc, tc), np.zeros(3))
            u = b1v - v; u = u / max(np.linalg.norm(u), 1e-9)
            e1 = np.cross(u, [0, 0, 1.0]); e1 = e1 / max(np.linalg.norm(e1), 1e-9); e2 = np.cross(u, e1)
            deltas = [np.zeros(3)] + [DMAX * r * (np.cos(th) * e1 + np.sin(th) * e2) for r in (0.5, 1.0) for th in np.arange(0, 2 * np.pi, np.pi / 4)]
            ths = tc * np.linspace(1.0, 2.5, 31)
            opt0 = min(two_leg(p, v, b1p, b1v, b2p, b2v, a, th, np.zeros(3)) for th in ths)
            opt = min(two_leg(p, v, b1p, b1v, b2p, b2v, a, th, d) for th in ths for d in deltas)
            rows.append(dict(name=n, k=k, T1=t1 - t0, T2=t2_ - t1, base=base, opt0=opt0, opt=opt, tc=tc))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    for kind in ("their", "ours_strict", "ours_free"):
        df = run(kind)
        df = df[np.isfinite(df.base) & np.isfinite(df.opt)]
        s0, s = df.base - df.opt0, df.base - df.opt
        act = df.T1 + df.T2
        print(f"{kind:12s} n={len(df)} two-leg actual {act.median():.2f}  model base {df.base.median():.2f}  opt(t_h) {df.opt0.median():.2f}  "
              f"opt(t_h, delta<={DMAX}) {df.opt.median():.2f} | saved per 2 legs: t_h only {s0.mean():.2f} [{s0.median():.2f}], "
              f"+aim {s.mean():.2f} [{s.median():.2f}]")

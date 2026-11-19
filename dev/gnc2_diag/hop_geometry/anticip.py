"""Anticipation: around each pop k (t_k + tau, tau in [-1.5, +1.0] s), angle between the measured thrust direction and
 (a) the min-time force direction for the NEXT balloon b_{k+1} computed from the state at that moment,
 (b) the min-time force direction for the CURRENT balloon b_k (before the pop).
Also the rocket's miss distance to b_k if it coasted from t_k + tau with thrust = -g (straight line, relative).
Median over hops, their vs ours_strict vs ours_free."""
import numpy as np, pandas as pd
from pathlib import Path
import hops as H
here = Path(__file__).parent
TAUS = np.round(np.arange(-1.5, 1.01, 0.1), 2)


def run(kind):
    rec = []
    for n in H.NAMES:
        z = np.load(H.D / f"{kind}_{n}.npz")
        t, R, B, pops = z["t"], z["R"], z["B"], z["pops"]
        launch = float(z["launch"]); live = ~np.isnan(R[:, 0]); idx = np.flatnonzero(live)
        fa = np.full_like(R[:, :3], np.nan); fa[idx] = H.thrust_acc(t[idx], R[idx]); fm = np.linalg.norm(fa, axis=1)
        burn = live & (t > launch + 0.5) & (t < launch + 29.5); a_cap = float(np.nanpercentile(fm[burn], 95))
        P = [(float(tp), int(b)) for tp, b in pops if tp <= launch + H.BURN]
        for k in range(len(P) - 1):
            (t0, b0), (t1, b1) = P[k], P[k + 1]
            prev_t = P[k - 1][0] if k > 0 else launch
            i0 = int(np.argmin(np.abs(t - t0)))
            for tau in TAUS:
                if t0 + tau <= prev_t + 0.05 or t0 + tau >= t1 - 0.05:
                    continue
                j = i0 - 1 + int(round(tau / 0.01))
                if not live[j] or t[j] > launch + 29.5:
                    continue
                p, v, f = R[j, :3], R[j, 3:6], fa[j]
                _, fn = H.tmin(p, v, B[j, b1, :3].astype(float), B[j, b1, 3:].astype(float), a_cap)
                row = dict(kind=kind, name=n, k=k, tau=tau, a_next=H.ang(f, fn), fmag=fm[j] / a_cap)
                if tau < 0:
                    bp, bv = B[j, b0, :3].astype(float), B[j, b0, 3:].astype(float)
                    _, fc = H.tmin(p, v, bp, bv, a_cap)
                    row["a_cur"] = H.ang(f, fc)
                    r, vr = bp - p, bv - v  # closest approach coasting with gravity cancelled
                    tc = max(-(r @ vr) / max(vr @ vr, 1e-9), 0.0)
                    row["zem_cur"] = np.linalg.norm(r + vr * tc); row["tgo_cur"] = tc
                rec.append(row)
    return pd.DataFrame(rec)


if __name__ == "__main__":
    df = pd.concat([run(k) for k in ("their", "ours_strict", "ours_free")])
    df.to_csv(here / "anticip.csv", index=False)
    pd.set_option("display.width", 250)
    t = df.groupby(["tau", "kind"])[["a_next", "a_cur", "zem_cur", "fmag"]].median().unstack("kind")
    print(t.round(1).to_string())

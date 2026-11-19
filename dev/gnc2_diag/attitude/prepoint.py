"""Do they pre-point? Around each pop k (not the last): the NEXT hop's hindsight direction evaluated at the pop,
f_next = 2 (pb_next(t_pop_next) - p(t_pop) - v(t_pop) tau) / tau^2 - G, and the angle of the nose to it over
t_pop - 1.0 .. t_pop + 0.5. Also the angle of the nose to the CURRENT balloon's hindsight direction in the last second
(if it grows before the pop, they turn away early).  Their vs ours_strict (same order) vs ours_free."""
import numpy as np
from common import *

OFFS = np.array([-1.0, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4])


def main():
    for kind in ("their", "ours_strict", "ours_free"):
        NX, CUR, VANG, WP = [], [], [], []
        for n in NAMES:
            d = load(kind, n)
            t, p, v, zb, B = d["t"], d["p"], d["v"], d["zb"], d["B"]
            H = hops(d)
            for (ts, te, b, first), (ts2, te2, b2, _) in zip(H[:-1], H[1:]):
                if first:
                    continue
                i = idx(t, te)
                tau = te2 - te
                pb2 = B[max(idx(t, te2) - 1, 0), b2, :3].astype(float)
                fn = 2 * (pb2 - p[i] - v[i] * tau) / tau ** 2 - G
                row = []
                for o in OFFS:
                    j = idx(t, te + o)
                    row.append(np.degrees(ang(zb[j], fn)) if d["live"][j] and ts <= te + o else np.nan)
                NX.append(row)
                # current balloon, last second before the pop (tau_cur >= 0.3 only)
                pb = B[max(i - 1, 0), b, :3].astype(float)
                rc = []
                for o in (-1.0, -0.6, -0.4, -0.3):
                    j = idx(t, te + o); tc = -o
                    fc = 2 * (pb - p[j] - v[j] * tc) / tc ** 2 - G
                    rc.append(np.degrees(ang(zb[j], fc)) if ts <= te + o else np.nan)
                CUR.append(rc)
                # velocity direction at the pop vs direction to the next balloon at its pop time
                VANG.append(np.degrees(ang(v[i], pb2 - p[i])))
                # body rate at the pop and whether the nose is turning toward (-) or away (+) from the next direction
                j1 = min(i + 5, len(t) - 1); j0 = max(i - 5, 0)
                dth = (np.degrees(ang(zb[j1], fn)) - np.degrees(ang(zb[j0], fn))) / (t[j1] - t[j0])
                WP.append((d["wxy"][i], np.max(d["wxy"][max(i - 50, 0):i + 1]), dth))
        NX, CUR = np.array(NX), np.array(CUR)
        print(f"[{kind}] n={len(NX)}  angle(nose, next-hop direction at pop) median at t_pop + "
              + "  ".join(f"{o:+.1f}s:{np.nanmedian(NX[:, k]):5.1f}" for k, o in enumerate(OFFS)))
        print(f"        angle(nose, current-balloon hindsight dir) median at t_pop " +
              "  ".join(f"{o:+.1f}s:{np.nanmedian(CUR[:, k]):5.1f}" for k, o in enumerate((-1.0, -0.6, -0.4, -0.3))))
        WP = np.array(WP)
        print(f"        |w| at pop median {np.median(WP[:, 0]):.2f} rad/s (p90 {np.percentile(WP[:, 0], 90):.2f}); max |w| in last 0.5 s median {np.median(WP[:, 1]):.2f};"
              f" d/dt angle(nose, next dir) at pop median {np.median(WP[:, 2]):+.1f} deg/s, turning away (>+20 deg/s) in {np.mean(WP[:, 2] > 20):.2f} of pops")
        print(f"        angle(velocity at pop, next balloon position at its pop - rocket pos) median {np.median(VANG):.1f} deg")


if __name__ == "__main__":
    main()

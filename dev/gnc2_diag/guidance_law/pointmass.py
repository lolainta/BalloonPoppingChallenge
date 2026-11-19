"""The guidance law alone (no attitude dynamics, no drag): point mass, thrust accel a = 13 m/s^2 in any direction instantly,
gravity; balloon at constant velocity (4.4, -0.6, 6.1). Our law: a_cmd = N (r + vr t_go) / t_go^2, t_go = max(s * T1, 0.3)
with T1 = min reach time (centre, tol 0) re-estimated every step, then _saturate (gravity first, |f| <= a). Random
geometries like our hop starts (range 15-60 m, rocket speed 10-25 m/s in a random direction). Reports realized time / T1_0
per stretch level, hit rate (within 1.5 m before 3 * T1_0), arrival relative speed.
Also the same with an attitude-rate limit on the thrust direction (2.0 rad/s) to show what the turn costs."""
import numpy as np

G = np.array([0.0, 0.0, -9.81]); A = 13.0; DT = 0.01; GRID = np.arange(0.02, 15.0, 0.02)
VB = np.array([4.4, -0.6, 6.1])


def T1(r, vr, tol=0.0):
    t = GRID[:, None]
    m = r[None] + vr[None] * t - 0.5 * G[None] * t ** 2
    ok = np.linalg.norm(m, axis=1) <= 0.5 * A * GRID ** 2 + tol
    return GRID[np.argmax(ok)] if ok.any() else np.inf


def saturate(a):
    g = -G; a = a.copy(); a[2] = max(a[2], 1.0 - g[2])
    AA, BB, CC = a @ a, 2 * a @ g, g @ g - A ** 2
    s = 1.0
    if AA > 1e-9 and (a + g) @ (a + g) > A ** 2:
        s = max((-BB + np.sqrt(max(BB * BB - 4 * AA * CC, 0.0))) / (2 * AA), 0.0)
    return s * a + g


def fly(r0, v0, s, N=2.0, rate=None, f0=None):
    p, v, b = np.zeros(3), v0.copy(), r0.copy()
    t0 = T1(b - p, VB - v); fd = f0 if f0 is not None else np.array([0, 0, 1.0])
    t = 0.0
    while t < 3 * t0 + 1:
        r, vr = b - p, VB - v
        tt = T1(r, vr)
        tg = max(s * (tt if np.isfinite(tt) else 15.0), 0.3)
        f = saturate(N * (r + vr * tg) / tg ** 2)
        if rate is not None:  # thrust direction turns at most rate * DT per step
            u = f / np.linalg.norm(f); c = np.clip(fd @ u, -1, 1); th = np.arccos(c)
            if th > rate * DT:
                ax = u - c * fd; ax /= max(np.linalg.norm(ax), 1e-12)
                u = np.cos(rate * DT) * fd + np.sin(rate * DT) * ax
            fd = u; f = u * np.linalg.norm(f)
        pn = p + v * DT + 0.5 * (f + G) * DT ** 2; vn = v + (f + G) * DT; bn = b + VB * DT
        # swept closest approach
        r0_, r1_ = b - p, bn - pn; dr = r1_ - r0_
        k = np.clip(-(r0_ @ dr) / max(dr @ dr, 1e-12), 0, 1)
        if np.linalg.norm(r0_ + k * dr) <= 1.5:
            return t + k * DT, t0, np.linalg.norm(vn - VB)
        p, v, b, t = pn, vn, bn, t + DT
    return np.nan, t0, np.nan


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    cases = []
    for _ in range(120):
        d = rng.normal(size=3); d /= np.linalg.norm(d); d[2] = abs(d[2]) * 0.6; d /= np.linalg.norm(d)
        r0 = d * rng.uniform(15, 60)
        u = rng.normal(size=3); u /= np.linalg.norm(u)
        v0 = u * rng.uniform(10, 25)
        cases.append((r0, v0, u))
    for rate in (None, 2.0):
        print(f"--- thrust-direction rate limit: {rate} rad/s (initial thrust along the velocity)")
        for s in (0.6, 0.8, 1.0, 1.3):
            res = np.array([fly(r0, v0, s, rate=rate, f0=u) for r0, v0, u in cases])
            ok = np.isfinite(res[:, 0])
            ratio = res[ok, 0] / res[ok, 1]
            print(f"s={s}: hit {ok.mean():.2f}  realized/T1_0 median {np.median(ratio):.2f} mean {ratio.mean():.2f}  "
                  f"realized/(s*T1_0) median {np.median(res[ok, 0] / (s * res[ok, 1])):.2f}  vrel at hit median {np.nanmedian(res[:, 2]):.1f}")

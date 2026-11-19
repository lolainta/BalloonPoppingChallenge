"""Own point-mass check of the guidance law alone (instant thrust direction, |f| <= a, gravity, no drag, static balloon):
t_go = s * T1c (T1c = centre-hit min time recomputed every step, as the GNC), f = sat(N ZEM / t_go^2 + g). Time to within
1.5 m vs the ideal T1 (1.5 m tolerance) from the initial state. 150 random geometries."""
import numpy as np
G = np.array([0, 0, -9.81]); A = 13.0; dt = 0.01
grid = np.arange(0.01, 20, 0.01)


def T1(r, vr, tol):
    miss = r[None] + vr[None] * grid[:, None] - 0.5 * G * grid[:, None] ** 2
    ok = np.linalg.norm(miss, axis=1) <= 0.5 * A * grid ** 2 + tol
    return grid[np.argmax(ok)] if ok.any() else np.inf


def sat(a):
    g = -G; a = a.copy(); a[2] = max(a[2], 1.0 - g[2])
    AA, B, C = a @ a, 2 * a @ g, g @ g - A ** 2
    s = 1.0
    if AA > 1e-9 and (a + g) @ (a + g) > A ** 2:
        s = max((-B + np.sqrt(max(B * B - 4 * AA * C, 0))) / (2 * AA), 0)
    return s * a + g


def fly(r, vr, s, N=2.0):
    # r = balloon - rocket, vr = balloon vel - rocket vel (balloon static => vr = -v)
    t = 0.0
    while t < 25:
        if np.linalg.norm(r) <= 1.5: return t
        tg = T1(r, vr, 0.0); tg = 20 if not np.isfinite(tg) else tg
        tg = max(s * tg, 0.3)
        f = sat(N * (r + vr * tg) / tg ** 2)
        acc = f + G
        r = r - (-vr) * dt - 0.5 * acc * dt ** 2  # rocket moves: r decreases by v dt + a dt^2/2
        vr = vr - acc * dt
        t += dt
    return np.inf


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    res = {s: [] for s in (0.8, 1.0, 1.3)}
    for _ in range(150):
        d = rng.normal(size=3); d /= np.linalg.norm(d); r = d * rng.uniform(15, 60)
        v = rng.normal(size=3); v = v / np.linalg.norm(v) * rng.uniform(5, 25)
        t_id = T1(r, -v, 1.5)
        for s in res:
            res[s].append(fly(r.copy(), -v.copy(), s) / t_id)
    for s, x in res.items():
        x = np.array(x); x = x[np.isfinite(x)]
        print(f's={s}: actual/T1(1.5 m) median {np.median(x):.2f} mean {np.mean(x):.2f} p90 {np.percentile(x, 90):.2f} (n {len(x)})')

    # same, only geometries with the balloon above the rocket and closing velocity (closer to the flight distribution)
    res = {s: [] for s in (0.8, 1.0, 1.3)}
    rng = np.random.default_rng(2)
    k = 0
    while k < 100:
        d = rng.normal(size=3); d /= np.linalg.norm(d)
        if d[2] < 0.1: continue
        r = d * rng.uniform(15, 60)
        v = rng.normal(size=3); v = v / np.linalg.norm(v) * rng.uniform(5, 25)
        if v @ d < 0: continue
        k += 1
        t_id = T1(r, -v, 1.5)
        for s in res:
            res[s].append(fly(r.copy(), -v.copy(), s) / t_id)
    for s, x in res.items():
        x = np.array(x); x = x[np.isfinite(x)]
        print(f'above+closing  s={s}: actual/T1(1.5 m) median {np.median(x):.2f} mean {np.mean(x):.2f} p90 {np.percentile(x, 90):.2f} (n {len(x)})')

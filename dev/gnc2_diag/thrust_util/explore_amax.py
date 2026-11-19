"""Empirical max thrust acceleration a_max(tau), tau = time since launch, from OUR logs where throttle = 1."""
import numpy as np, glob
G = np.array([0, 0, -9.81])
K_DRAG = 0.5 * 0.25 * np.pi * 0.22**2  # N per (kg/m^3 (m/s)^2); divide by mass
taus, accs = [], []
for f in sorted(glob.glob('data/gnc2/ours_*.npz')):
    d = np.load(f); t, R, A = d['t'], d['R'], d['A']; L = float(d['launch'])
    v = R[:, 3:6]; dv = np.full_like(v, np.nan); dv[1:-1] = (v[2:] - v[:-2]) / 0.02
    fsp = dv - G
    tau = t - L
    gx, gy = np.radians(A[:, 0]), np.radians(A[:, 1])
    cosf = np.sqrt(1 - np.sin(gx)**2 - np.sin(gy)**2)
    thr = A[:, 3]
    ok = (tau > 0.5) & (tau < 29.9) & (thr > 0.999) & np.isfinite(fsp).all(1)
    ok[1:] &= thr[:-1] > 0.999
    speed = np.linalg.norm(v, axis=1)
    drag = K_DRAG * 1.2 * speed**2 / 80.0
    # thrust accel magnitude ~ |f| + drag component (drag opposes v; thrust mostly not aligned with v -> approximate)
    fm = np.linalg.norm(fsp, axis=1)
    taus.append(tau[ok]); accs.append((fm / cosf)[ok])
taus = np.concatenate(taus); accs = np.concatenate(accs)
for a, b in [(0.5, 2), (2, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 29.9)]:
    m = (taus >= a) & (taus < b)
    print(f"tau {a:5.1f}-{b:5.1f}: n={m.sum():6d} |f|/cos p10 {np.percentile(accs[m],10):6.2f} p50 {np.percentile(accs[m],50):6.2f} p90 {np.percentile(accs[m],90):6.2f}")
# model: T / (m0 - mdot tau)
m0 = 70 + 13.9 + 2.13 + np.pi*(0.0843**2-0.0295**2)*0.2757*900
print('m0', m0, 'a0', 1080/m0)

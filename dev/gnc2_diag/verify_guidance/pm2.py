"""pm.py variant: same law, report both denominators (T1 with 1.5 m and centre T1) and the fraction of flights that take
longer than 3 * T1c + 1 s (which a 'hit before 3 T1' criterion would drop from the median)."""
import numpy as np
from pm import T1, fly
rng = np.random.default_rng(3)
cases = []
while len(cases) < 100:
    d = rng.normal(size=3); d /= np.linalg.norm(d); d[2] = abs(d[2]) * 0.6; d /= np.linalg.norm(d)
    r = d * rng.uniform(15, 60); u = rng.normal(size=3); u /= np.linalg.norm(u); v = u * rng.uniform(10, 25)
    cases.append((r, v))
for s in (0.8, 1.0, 1.3):
    a15, a0, late = [], [], 0
    for r, v in cases:
        t = fly(r.copy(), -v.copy(), s); t15, t0 = T1(r, -v, 1.5), T1(r, -v, 0.0)
        if not np.isfinite(t) or t > 3 * t0 + 1: late += 1
        a15.append(t / t15); a0.append(t / t0)
    a15, a0 = np.array(a15), np.array(a0)
    keep = np.isfinite(a0) & (a0 <= 3 + 1 / 1.0)
    print(f's={s}: /T1(1.5) med {np.nanmedian(a15):.2f} | /T1c med {np.nanmedian(a0):.2f} | late/miss (>3 T1c+1) {late} of {len(cases)} | /T1c med excluding those {np.median(a0[keep]):.2f}')

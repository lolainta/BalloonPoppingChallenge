"""Grid-search a near-time-optimal planar attitude law on the real hardware limits (alpha = K a sin g, 15 deg,
60 deg/s slew, one-step delay): settle time (last time |error| > 5 deg) averaged over 20..135 deg steps,
overshoot <= 5 deg. Also reports our current law."""
import itertools
import numpy as np
from common import K_ROT

def sim(Theta, law, a_thr=13.0, T=3.0, dt=0.01):
    th = w = g = gp = 0.0
    out = []
    kind, p = law
    for i in range(int(T / dt)):
        e = Theta - th
        if kind == "lin":
            wc = np.sign(e) * min(p["mr"], p["k"] * abs(e))
        else:
            wc = np.sign(e) * min(p["mr"], np.sqrt(2 * p["ad"] * abs(e)), p["k"] * abs(e))
        gc = np.clip(p["kr"] * (wc - w), -15, 15)
        g = g + np.clip(gp - g, -0.6, 0.6); gp = gc
        w += K_ROT * a_thr * np.sin(np.radians(g)) * dt; th += w * dt
        out.append(th)
    e = np.degrees(Theta - np.array(out))
    bad = np.flatnonzero(np.abs(e) > 5)
    ts = (bad[-1] + 1) * dt if len(bad) else 0.0
    t10 = (np.argmax(e < 10) + 1) * dt
    return ts, t10, max(0.0, -e.min())

ANG = np.radians([20, 35, 50, 90, 135])
def score(law):
    r = [sim(a, law) for a in ANG]
    return np.mean([x[0] for x in r]), np.mean([x[1] for x in r]), max(x[2] for x in r), r

best = []
ours = ("lin", dict(mr=2.5, k=3.0, kr=10.0))
s = score(ours); print("ours", "settle %.3f t10 %.3f overshoot %.1f" % s[:3], [(round(a, 2), round(b, 2)) for a, b, _ in s[3]])
for k, mr, kr in itertools.product([3, 4, 5, 6, 8], [2.5, 3, 4, 5], [6, 8, 10, 12, 15, 20]):
    law = ("lin", dict(mr=mr, k=k, kr=kr)); s = score(law)
    if s[2] <= 5: best.append((s[0], s[1], s[2], law))
for ad, k, mr, kr in itertools.product([1.5, 2, 2.5, 3, 4, 5], [4, 6, 8, 10], [2.5, 3, 4, 5], [6, 8, 10, 12, 15, 20]):
    law = ("sqrt", dict(ad=ad, mr=mr, k=k, kr=kr)); s = score(law)
    if s[2] <= 5: best.append((s[0], s[1], s[2], law))
best.sort(key=lambda x: x[0])
for b in best[:8]:
    print("settle %.3f t10 %.3f overshoot %.1f" % b[:3], b[3], [(round(a, 2), round(c, 2)) for a, c, _ in score(b[3])[3]])

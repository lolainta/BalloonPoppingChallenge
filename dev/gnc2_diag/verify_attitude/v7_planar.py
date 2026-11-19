"""V7: planar rotation from rest with our cascade; delay 0 or 1 step; a = 13 m/s^2, K = 2.30; slew 0.6 deg/step."""
import numpy as np

def run(Theta_deg, k_att=3.0, max_rate=2.5, k_rate=10.0, delay=0, a=13.0, K=2.30, T=4.0, law="lin", a_dec=2.5, k_lin=6.0):
    Th = np.radians(Theta_deg); th = w = g = 0.0; pend = [0.0] * delay
    out = []
    for i in range(int(T / 0.01)):
        e = Th - th
        if law == "lin": wc = np.sign(e) * min(max_rate, k_att * abs(e))
        else: wc = np.sign(e) * min(max_rate, np.sqrt(2 * a_dec * abs(e)), k_lin * abs(e))
        gc = float(np.clip(k_rate * (wc - w), -15, 15))
        pend.append(gc); gc = pend.pop(0)
        g += np.clip(gc - g, -0.6, 0.6)
        w += K * a * np.sin(np.radians(g)) * 0.01; th += w * 0.01
        out.append(np.degrees(th))
    out = np.array(out); err = np.abs(Theta_deg - out)
    t10 = (np.argmax(err < 10) + 1) * 0.01
    bad = np.flatnonzero(err >= 5); ts5 = (bad[-1] + 2) * 0.01 if len(bad) else 0.01
    return t10, ts5, out.max() - Theta_deg

for delay in (0, 1):
    r = np.array([run(T, delay=delay) for T in np.arange(20, 136, 5)])
    t90 = run(90, delay=delay)
    print(f"ours delay {delay}: 20-135 mean t10 {r[:,0].mean():.2f} settle5 {r[:,1].mean():.2f} max overshoot {r[:,2].max():.1f} | 90deg t10 {t90[0]:.2f} settle {t90[1]:.2f}")
    r = np.array([run(T, delay=delay, law="sqrt", max_rate=4.0, k_rate=12.0, a_dec=2.5, k_lin=6.0) for T in np.arange(20, 136, 5)])
    t90 = run(90, delay=delay, law="sqrt", max_rate=4.0, k_rate=12.0, a_dec=2.5, k_lin=6.0)
    print(f"sqrt  delay {delay}: 20-135 mean t10 {r[:,0].mean():.2f} settle5 {r[:,1].mean():.2f} max overshoot {r[:,2].max():.1f} | 90deg t10 {t90[0]:.2f} settle {t90[1]:.2f}")
    for ka, mr, kr in ((4, 3, 10), (5, 4, 15), (3, 4, 10)):
        r = np.array([run(T, k_att=ka, max_rate=mr, k_rate=kr, delay=delay) for T in np.arange(20, 136, 5)])
        print(f"   lin k_att {ka} mr {mr} kr {kr}: t10 {r[:,0].mean():.2f} settle5 {r[:,1].mean():.2f} overshoot max {r[:,2].max():.1f}")

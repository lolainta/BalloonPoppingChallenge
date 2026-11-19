"""Path shape and chaining: sequence bend vs arrival alignment; thrust use; column climbing / fly-through."""
import pandas as pd, numpy as np
from pathlib import Path
here = Path(__file__).parent
pd.set_option("display.width", 250)


def ang(a, b):
    na, nb = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
    return np.degrees(np.arccos(np.clip((a * b).sum(-1) / np.maximum(na * nb, 1e-9), -1, 1)))


def prep(kind):
    h = pd.read_csv(here / f"hops_{kind}.csv")
    out = []
    for n, g in h.groupby("name"):
        g = g.sort_values("k").reset_index(drop=True)
        P = g[["px", "py", "pz"]].to_numpy(); V = g[["vx", "vy", "vzz"]].to_numpy(); Rl = g[["rx", "ry", "rz"]].to_numpy()
        chord = np.vstack([np.full(3, np.nan), P[1:] - P[:-1]])  # arriving chord (previous pop -> this pop)
        g["bend"] = ang(chord, Rl)              # sequence bend: arriving chord vs LOS to the next balloon
        g["v_vs_chord"] = ang(V, chord)         # arrival velocity vs arriving chord (path curvature at the end)
        out.append(g)
    h = pd.concat(out)
    return h[h.k > 0]


R = {k: prep(k) for k in ("their", "ours_strict", "ours_free")}
cols = ["bend", "v_vs_chord", "th_int", "el_los", "el_v", "vmin_hop", "speed", "speed1", "vturn", "dh", "dzp",
        "thr_use", "consist", "along", "t_turn", "nose_err"]
tab = pd.DataFrame({k: h[cols].median() for k, h in R.items()})
print("medians\n", tab.round(2).to_string())
for k, h in R.items():
    print(f"{k:12s} hops going up (next balloon above) {np.mean(h.dz > 0):.2f} | LOS elevation >30deg {np.mean(h.el_los > 30):.2f} | "
          f"hop rises > horizontal {np.mean(h.dzp > h.dh):.2f} | speed kept (vmin >= 0.8 speed at pop) {np.mean(h.vmin_hop >= 0.8 * h.speed):.2f} | "
          f"th_int<20: {np.mean(h.th_int < 20):.2f}  th_int<10: {np.mean(h.th_int < 10):.2f}")
# alignment vs bend: does good alignment come from the sequence (small bend) or from the approach (v off the chord)?
for k, h in R.items():
    c = np.corrcoef(h.bend.fillna(h.bend.median()), h.th_int)[0, 1]
    print(f"{k:12s} corr(bend, th_int) {c:.2f}; th_int - bend median {np.median(h.th_int - h.bend):+.1f} deg")
# paired
m = R["their"].merge(R["ours_strict"], on=["name", "b0", "b1"], suffixes=("_th", "_us"))
print("\npaired (same balloon pair), medians their / ours_strict / diff:")
for c in ["bend", "v_vs_chord", "th_int", "el_v", "vmin_hop", "thr_use", "consist", "along", "t_turn", "nose_err", "vturn"]:
    a, b = m[c + "_th"], m[c + "_us"]
    print(f"  {c:11s} {a.median():7.2f} {b.median():7.2f}  {np.nanmedian(b - a):+7.2f}")

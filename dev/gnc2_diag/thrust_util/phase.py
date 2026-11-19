"""Where in a hop is thrust magnitude lost? Utilization U binned by time since hop start and time to the pop,
plus our throttle-loss causes by time-to-pop. Hops = previous pop (or launch) -> next pop, burn samples only."""
import sys; sys.path.insert(0, 'dev/gnc2_diag/thrust_util')
import numpy as np
from util_analysis import load, derive, NAMES, DT
bins = [0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 99]
out = {}
for kind in ("their", "ours_strict", "ours_free"):
    since, togo, U, lost, speed0 = [], [], [], [], []
    thr, st, clamp = [], [], []
    for n in NAMES:
        d = load(kind, n); x = derive(d); t = x["t"]; m = x["burn"]
        ev = np.concatenate([[x["L"]], d["pops"][:, 0]])
        k = np.searchsorted(ev, t, side="right")  # hop index
        valid = m & (k >= 1) & (k < len(ev))
        s_ = t - ev[np.clip(k - 1, 0, len(ev) - 1)]; g_ = ev[np.clip(k, 0, len(ev) - 1)] - t
        since.append(s_[valid]); togo.append(g_[valid]); U.append(x["U"][valid]); lost.append(((1 - x["U"]) * x["am"])[valid])
        if "A" in d:
            thr.append(d["A"][valid, 3]); st.append(d["STRETCH"][valid])
            F = d["F"][valid]; Fm = np.linalg.norm(F, axis=1)
            clamp.append(np.isfinite(Fm) & (F[:, 2] < 1.05) & (Fm < 0.98 * d["a_thrust"][valid]))
    since, togo, U, lost = map(np.concatenate, (since, togo, U, lost))
    print(f"== {kind}: n={U.size}  total lost dv/run {lost.sum() * DT / 11:.1f} m/s")
    for lab, arr in (("since hop start", since), ("time to pop", togo)):
        row = []
        for a, b in zip(bins[:-1], bins[1:]):
            q = (arr >= a) & (arr < b)
            row.append(f"[{a:g},{b:g}) U={U[q].mean():.3f} f<.8={np.mean(U[q] < .8):.2f} lost={lost[q].sum() * DT / 11:5.1f}")
        print(f"  {lab}:"); [print("    " + r) for r in row]
    if thr:
        thr, st, clamp = map(np.concatenate, (thr, st, clamp))
        print("  our throttle by time-to-pop and speed level (mean throttle / share of burn):")
        for a, b in zip(bins[:-1], bins[1:]):
            q = (togo >= a) & (togo < b)
            cells = " ".join(f"s{L}:{thr[q & (np.abs(st - L) < 1e-6)].mean() if (q & (np.abs(st - L) < 1e-6)).any() else float('nan'):.2f}/{np.mean(q & (np.abs(st - L) < 1e-6)):.3f}"
                             for L in (0.8, 1.0, 1.3))
            print(f"    to-pop [{a:g},{b:g}) {cells}  clamp share {np.mean(q & clamp):.3f}")

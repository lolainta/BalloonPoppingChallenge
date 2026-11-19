"""t_go estimate quality along our strict chases: at each step the guidance flies t_go = STRETCH * TGO (TGO = its min reach
time), while the actual remaining time is t_pop - t. Reports, by the stretch level in force, the ratio actual / planned at
chase start and along the chase, and the slope d(planned)/dt (a consistent t_go falls at -1 s/s)."""
import pickle
import numpy as np, pandas as pd
from pathlib import Path
from chases import D, NAMES

OUT = Path(__file__).resolve().parent
rows, prof = [], []
seen = set()
for kind in ("ours_strict", "ours_free"):
    for n in NAMES:
        f = np.load(D / f"{kind}_{n}.npz")
        t, TGT, TGO, ST, pops = f["t"], f["TGT"], f["TGO"], f["STRETCH"], f["pops"]
        for tp, b in pops:
            b = int(b)
            i1 = int(np.searchsorted(t, tp - 1e-6))
            j = i1 - 1 if TGT[i1 - 1] == b else None
            if j is None:  # free mode: handed off before the pop -> find the last step targeting b
                w = np.flatnonzero(TGT[:i1] == b)
                if not w.size or t[i1] - t[w[-1]] > 1.5:
                    continue
                j = w[-1]
            i0 = j
            while i0 > 0 and TGT[i0 - 1] == b:
                i0 -= 1
            key = (kind, int(f["seed"]), b, round(float(tp), 2))
            dup = key in seen; seen.add(key)
            tau_p = ST[i0:j + 1] * TGO[i0:j + 1]; tau_a = tp - t[i0:j + 1]
            ok = np.isfinite(tau_p) & (TGO[i0:j + 1] < 14.8)
            if ok.sum() < 20:
                continue
            tt = t[i0:j + 1][ok]; tp_ = tau_p[ok]; ta_ = tau_a[ok]
            slope = np.polyfit(tt[: max(len(tt) // 2, 5)], tp_[: max(len(tt) // 2, 5)], 1)[0]  # first half of the chase
            rows.append(dict(kind=kind, name=n, dup=dup, b=b, dur=tp - t[i0], s0=ST[i0], s_mean=ST[i0:j + 1].mean(), TGO0=TGO[i0],
                             plan0=tau_p[0], ratio0=(tp - t[i0]) / tau_p[0], slope=slope,
                             ratio_mid=np.median(ta_[len(ta_) // 3: 2 * len(ta_) // 3] / tp_[len(ta_) // 3: 2 * len(ta_) // 3]),
                             tgo_slope=np.polyfit(tt[: max(len(tt) // 2, 5)], TGO[i0:j + 1][ok][: max(len(tt) // 2, 5)], 1)[0]))
            for q in np.linspace(0, 0.95, 20):
                m = int(q * (len(tt) - 1))
                prof.append(dict(kind=kind, dup=dup, s=ST[i0 + m] if True else 0, q=round(q, 2), r=ta_[m] / tp_[m], s0=ST[i0]))
r = pd.DataFrame(rows)
pickle.dump(r, open(OUT / "tgo_quality.pkl", "wb"))
pd.set_option("display.width", 200)
for kind in ("ours_strict", "ours_free"):
    x = r[(r.kind == kind) & ~r.dup]
    print(f"=== {kind} (dedup, n={len(x)})")
    x = x.assign(sbin=x.s0.round(1))
    print(x.groupby("sbin").agg(n=("dur", "size"), dur=("dur", "mean"), TGO0=("TGO0", "mean"), plan0=("plan0", "mean"),
                                act_over_plan0=("ratio0", "median"), act_over_plan_mid=("ratio_mid", "median"),
                                slope_plan=("slope", "median"), slope_TGO=("tgo_slope", "median")).round(2).to_string())
    print("all: act/plan at start median %.2f, dur/TGO0 median %.2f, planned-t_go slope median %.2f s/s, TGO slope %.2f" % (
        x.ratio0.median(), (x.dur / x.TGO0).median(), x.slope.median(), x.tgo_slope.median()))

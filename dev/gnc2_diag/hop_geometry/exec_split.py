"""Execution part of ours_strict hops: stretch level during the hop, nose error, turn time; sanity of the Tmin model."""
import pandas as pd, numpy as np
from pathlib import Path
here = Path(__file__).parent; D = here.parents[2] / "data/gnc2"
pd.set_option("display.width", 250)
for kind in ("their", "ours_strict", "ours_free"):
    h = pd.read_csv(here / f"hops_{kind}.csv"); h = h[h.k > 0]
    print(kind, "frac T < Tmin: %.2f  min T/Tmin %.2f" % ((h["T"] < h.Tmin).mean(), (h["T"] / h.Tmin).min()))
for kind in ("ours_strict", "ours_free"):
    h = pd.read_csv(here / f"hops_{kind}.csv"); h = h[h.k > 0].copy()
    st = []
    for _, r in h.iterrows():
        z = np.load(D / f"{kind}_{r['name']}.npz")
        t = z["t"]; s = (t >= r.t0) & (t < r.t0 + r["T"])
        st.append(np.median(z["STRETCH"][s]))
    h["stretch"] = st
    h["exec"] = h["T"] - h.Tmin
    h["ratio"] = h["T"] / h.Tmin
    print(f"\n{kind}: by stretch level during the hop")
    print(h.groupby("stretch")[["T", "Tmin", "exec", "ratio", "th_int", "nose_err"]].agg(["count", "median"]).round(2).to_string())

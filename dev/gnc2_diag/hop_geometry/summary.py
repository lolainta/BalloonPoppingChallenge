import pandas as pd, numpy as np
from pathlib import Path
here = Path(__file__).parent
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
cols = ["T", "d", "th_los", "th_int", "vclose", "vperp", "speed", "vrel_nb", "vrel_hit", "vz", "alt", "dz", "dv_int", "dv_net",
        "nose_err", "vel_nose", "Tmin", "Tmin_al", "Tmin_0", "Tmin_b", "a_cap"]
res = {}
for kind in ("their", "ours_strict", "ours_free"):
    h = pd.read_csv(here / f"hops_{kind}.csv")
    h = h[h.k > 0]
    res[kind] = h
    q = h[cols].quantile([0.25, 0.5, 0.75]).T
    q["mean"] = h[cols].replace(np.inf, np.nan).mean()
    print(f"\n=== {kind}: {len(h)} pop->pop hops ===")
    print(q.round(2).to_string())
# derived
for kind, h in res.items():
    eff = h["T"] / h["Tmin"]
    print(kind, "T/Tmin median %.2f  T-Tmin median %.2f mean %.2f | Tmin-Tmin_al median %.2f mean %.2f | frac th_int<45 %.2f  >90 %.2f | frac vclose>0 %.2f"
          % (eff.median(), (h["T"] - h["Tmin"]).median(), (h["T"] - h["Tmin"]).mean(), (h["Tmin"] - h["Tmin_al"]).median(),
             (h["Tmin"] - h["Tmin_al"]).mean(), (h.th_int < 45).mean(), (h.th_int > 90).mean(), (h.vclose > 0).mean()))

"""Paired hops: the same balloon pair (b0 -> b1), popped consecutively by them and by ours_strict."""
import pandas as pd, numpy as np
from pathlib import Path
here = Path(__file__).parent
pd.set_option("display.width", 250)
th = pd.read_csv(here / "hops_their.csv"); us = pd.read_csv(here / "hops_ours_strict.csv")
th, us = th[th.k > 0], us[us.k > 0]
m = th.merge(us, on=["name", "b0", "b1"], suffixes=("_th", "_us"))
print("paired hops:", len(m), "of their", len(th), "/ ours", len(us))
cols = ["T", "d", "th_int", "vclose", "vperp", "speed", "vrel_nb", "vz", "nose_err", "Tmin", "Tmin_al", "Tmin_b", "dv_int", "tb_left"]
rows = []
for c in cols:
    a, b = m[c + "_th"].replace(np.inf, np.nan), m[c + "_us"].replace(np.inf, np.nan)
    dd = (b - a)
    rows.append([c, a.median(), b.median(), dd.median(), dd.mean(), dd.std() / np.sqrt(dd.notna().sum())])
print(pd.DataFrame(rows, columns=["metric", "their_med", "ours_med", "diff_med", "diff_mean", "diff_se"]).round(2).to_string(index=False))
# decomposition of the hop-time gap on paired hops: T = Tmin + (T - Tmin); Tmin = Tmin_al + (Tmin - Tmin_al)
for lab, c in [("T", None)]:
    pass
dT = (m.T_us - m.T_th)
geo_align = (m.Tmin_us - m.Tmin_al_us) - (m.Tmin_th - m.Tmin_al_th)
geo_speed = m.Tmin_al_us - m.Tmin_al_th
execu = (m.T_us - m.Tmin_us) - (m.T_th - m.Tmin_th)
print("\nPaired decomposition of the hop-time gap (ours - their), mean [median]:")
for lab, x in [("total dT", dT), ("alignment at pop (Tmin - Tmin_al)", geo_align), ("speed/distance at pop (Tmin_al)", geo_speed), ("execution (T - Tmin)", execu)]:
    print(f"  {lab:40s} {x.mean():+.2f} [{x.median():+.2f}]  se {x.std()/np.sqrt(len(x)):.2f}")
m.to_csv(here / "paired.csv", index=False)

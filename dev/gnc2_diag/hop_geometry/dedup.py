"""Headline hop numbers with duplicate flights removed: ours_strict 13_Apex == 12_Apex and 16_SpaceBerry == 18_Taoyuan
(same seed, same launch decision step); ours_free has one flight per seed (5 unique: seeds 0, 14, 1362, 1957083, 205)."""
import pandas as pd, numpy as np
from pathlib import Path
here = Path(__file__).parent
DUP_STRICT = {"13_Apex", "16_SpaceBerry"}
FREE_KEEP = {"10_CoolBa", "10_SpaceY", "11_near_miss", "12_Apex", "14_SpaceBerry"}
th = pd.read_csv(here / "hops_their.csv"); us = pd.read_csv(here / "hops_ours_strict.csv"); fr = pd.read_csv(here / "hops_ours_free.csv")
th, us, fr = th[th.k > 0], us[(us.k > 0) & ~us.name.isin(DUP_STRICT)], fr[(fr.k > 0) & fr.name.isin(FREE_KEEP)]
cols = ["T", "d", "th_int", "vclose", "vperp", "speed", "vrel_hit", "nose_err", "t_turn", "vturn", "vmin_hop", "Tmin", "Tmin_al"]
print(pd.DataFrame({"their": th[cols].median(), "ours_strict(dedup)": us[cols].median(), "ours_free(dedup)": fr[cols].median()}).round(2).to_string())
print("n hops", len(th), len(us), len(fr))
m = th.merge(us, on=["name", "b0", "b1"], suffixes=("_th", "_us"))
dT = m.T_us - m.T_th
al = (m.Tmin_us - m.Tmin_al_us) - (m.Tmin_th - m.Tmin_al_th)
sp = m.Tmin_al_us - m.Tmin_al_th
ex = (m.T_us - m.Tmin_us) - (m.T_th - m.Tmin_th)
print(f"paired dedup n={len(m)}: dT {dT.mean():+.2f} [{dT.median():+.2f}] = alignment {al.mean():+.2f} [{al.median():+.2f}] + speed {sp.mean():+.2f} "
      f"[{sp.median():+.2f}] + execution {ex.mean():+.2f} [{ex.median():+.2f}]  (se of mean: {dT.std()/np.sqrt(len(m)):.2f}, {al.std()/np.sqrt(len(m)):.2f}, {sp.std()/np.sqrt(len(m)):.2f}, {ex.std()/np.sqrt(len(m)):.2f})")

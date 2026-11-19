import pandas as pd, numpy as np, sys
from pathlib import Path
here = Path(__file__).parent
tag = sys.argv[1] if len(sys.argv) > 1 else ""
DUP = {"13_Apex", "16_SpaceBerry"}; FREE = {"10_CoolBa", "10_SpaceY", "11_near_miss", "12_Apex", "14_SpaceBerry"}
th = pd.read_csv(here / "vhops_their.csv"); us = pd.read_csv(here / f"vhops_ours_strict{tag}.csv") ; fr = pd.read_csv(here / "vhops_ours_free.csv")
usall = us.copy(); us = us[~us.name.isin(DUP)]; frall = fr.copy(); fr = fr[fr.name.isin(FREE)]
cols = ["T", "d", "th", "th_ground", "vclose", "vperp", "speed", "vrel_hit", "vmin_ratio", "vturn", "dv_int", "fuse", "cap", "Tmin", "Tmin_al", "nose", "t_turn", "el_v", "alt"]
tab = pd.DataFrame({"their": th[cols].median(), "strict": us[cols].median(), "strict_all": usall[cols].median(), "free": fr[cols].median(), "free_all": frall[cols].median()})
tab.loc["mean_T"] = [th["T"].mean(), us["T"].mean(), usall["T"].mean(), fr["T"].mean(), frall["T"].mean()]
tab.loc["n"] = [len(th), len(us), len(usall), len(fr), len(frall)]
for lab, h in [("their", th), ("strict", us), ("strict_all", usall), ("free", fr), ("free_all", frall)]:
    tab.loc["frac_th<20", lab] = (h.th < 20).mean()
    tab.loc["frac_keep80", lab] = (h.vmin_ratio >= 0.8).mean()
    tab.loc["frac_up", lab] = (h.dz > 0).mean()
    tab.loc["frac_climb>horiz", lab] = (h.dzp > h.dh).mean()
    tab.loc["frac_d<15", lab] = (h.d < 15).mean()
    tab.loc["frac_nextalready_popped?", lab] = (h.st_next != 1).mean()
print(tab.round(3).to_string())

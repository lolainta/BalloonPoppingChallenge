"""Sensitivity of the 'sequence bend vs approach rotation' split: with/without the launch->first-pop chord, dedup vs all."""
import numpy as np, pandas as pd
from pathlib import Path
import vhops as V
here = Path(__file__).parent
DUP = {"13_Apex", "16_SpaceBerry"}; FREE = {"10_CoolBa", "10_SpaceY", "11_near_miss", "12_Apex", "14_SpaceBerry"}
def launch_pos(kind, name):
    z = np.load(V.D / f"{kind}_{name}.npz"); R = z["R"]; i = int(np.argmax(~np.isnan(R[:, 0]))); return R[i, :3]
def bend(kind, h, use_launch):
    out = []
    for n, g in h.groupby("name"):
        g = g.sort_values("k"); P = g[["px", "py", "pz"]].values; Rl = g[["rx", "ry", "rz"]].values
        prev = np.vstack([launch_pos(kind, n)[None], P[:-1]]) if use_launch else np.vstack([np.full((1, 3), np.nan), P[:-1]])
        for i in range(len(g)):
            if np.isnan(prev[i, 0]): continue
            out.append((V.angd(P[i] - prev[i], Rl[i]), g.th.values[i]))
    a = np.array(out); return a
th = pd.read_csv(here / "vhops_their.csv"); us = pd.read_csv(here / "vhops_ours_strict.csv"); fr = pd.read_csv(here / "vhops_ours_free.csv")
for use_launch in (True, False):
    for lab, kind, h in [("their", "their", th), ("strict_dedup", "ours_strict", us[~us.name.isin(DUP)]), ("strict_all", "ours_strict", us),
                         ("free_dedup", "ours_free", fr[fr.name.isin(FREE)]), ("free_all", "ours_free", fr)]:
        a = bend(kind, h, use_launch)
        print(f"launch_chord={use_launch!s:5} {lab:13s} n {len(a):3d} bend med {np.median(a[:,0]):5.1f} mean {a[:,0].mean():5.1f} | th med {np.median(a[:,1]):5.1f} "
              f"| th-bend med {np.median(a[:,1]-a[:,0]):+5.1f} mean {np.mean(a[:,1]-a[:,0]):+5.1f}")

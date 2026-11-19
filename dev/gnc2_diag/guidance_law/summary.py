"""Summaries of chases.pkl: hop time decomposition (T1 = ideal min time at chase start, excess = dur - T1), second passes,
stretch effect, dedup of the duplicated seeds (1957083: 12_Apex/13_Apex/16_SpaceBerry/18_Taoyuan; 0: CoolBa/Mochi/Yaka/TeamDJ)."""
import pickle
import numpy as np, pandas as pd
from pathlib import Path

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
r = pd.DataFrame(pickle.load(open(Path(__file__).with_name("chases.pkl"), "rb")))
seed = {"10_CoolBa": 0, "10_SpaceY": 14, "11_Mochi": 0, "11_near_miss": 1362, "12_Apex": 1957083, "13_Apex": 1957083, "13_Yaka": 0,
        "14_SpaceBerry": 205, "16_SpaceBerry": 1957083, "16_TeamDJ": 0, "18_Taoyuan": 1957083}
r["seed"] = r.name.map(seed)
r["excess"] = r.dur - r.T1
r["ratio"] = r.dur / r.T1
# a hop = a chase that starts right after one of our pops (k >= 1 for theirs) and ends with the pop
pt = {}
for (kind, name), g in r.groupby(["kind", "name"]):
    pt[(kind, name)] = np.sort(g.t1[g.popped].values)
def after_pop(row):
    p = pt[(row.kind, row["name"])]
    return bool(np.any(np.abs(p - (row.t0 - 0.01)) < 0.015))
r["after_pop"] = r.apply(after_pop, axis=1)
dd = r.drop_duplicates(["kind", "seed", "b", "t0", "t1"])
for label, df in (("ALL (11 replays)", r), ("DEDUP (identical chases removed)", dd)):
    print(f"==== {label}")
    for kind in ("their", "ours_strict", "ours_free"):
        x = df[(df.kind == kind) & df.popped & df.after_pop & np.isfinite(df.T1)]
        clean = x[x.nmiss == 0]
        print(f"{kind:12s} hops {len(x):3d}  dur med {x.dur.median():.2f} mean {x.dur.mean():.2f} | T1 med {x.T1.median():.2f} mean {x.T1.mean():.2f}"
              f" | excess med {x.excess.median():.2f} mean {x.excess.mean():.2f} | ratio med {x.ratio.median():.2f}"
              f" | miss-pass hops {int((x.nmiss > 0).sum())} ({(x.nmiss > 0).mean() * 100:.0f}%), lost sum {x.lost.sum():.1f}s = {x.lost.sum() / len(x):.2f}s/hop"
              f" | clean excess mean {clean.excess.mean():.2f} med {clean.excess.median():.2f}  vrel med {x.vrel.median():.1f}")

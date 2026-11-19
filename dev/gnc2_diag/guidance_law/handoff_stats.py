"""Handoff outcomes in a sim variant: switches away from a still-alive target after launch + first pop; popped within
1 s after the switch (coasted through) vs not (missed -> returned later?). Lead = pop time - switch time."""
import sys
import numpy as np
from chases import NAMES
from sim_report import load, FREE_NAMES

for v in sys.argv[1:]:
    names = FREE_NAMES if v.startswith("f") else NAMES
    st = dict(switch=0, coasted=0, missed=0, returned=0); lead = []
    for n in names:
        f = load(v, n); t, TGT = f["t"], f["TGT"]; pops = {int(b): tp for tp, b in f["pops"]}
        if not len(f["pops"]):
            continue
        t_first = f["pops"][0][0]
        for c in np.flatnonzero(np.diff(TGT) != 0):
            a, b = int(TGT[c]), int(TGT[c + 1]); tc = t[c + 1]
            if a < 0 or b < 0 or tc < t_first or pops.get(a, np.inf) <= tc + 0.005:
                continue
            st["switch"] += 1
            tp = pops.get(a, np.inf)
            if tp < tc + 1.0:
                st["coasted"] += 1; lead.append(tp - tc)
            else:
                st["missed"] += 1; st["returned"] += int((TGT[c + 1:] == a).any())
    print(v, st, "lead p10/50/90", np.round(np.percentile(lead, [10, 50, 90]), 2) if lead else None)

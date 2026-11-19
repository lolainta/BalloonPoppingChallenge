"""End of burn: after our last pop, which target were we chasing, was it reachable before burnout (TGO * stretch vs burn
left), and how close did we get (min distance to that target and to any live balloon until ground)?
.venv/bin/python dev/gnc2_diag/launch_regime/tail.py
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BURN, NAMES, load  # noqa: E402

for k in ("ours_strict", "ours_free"):
    print(f"--- {k}")
    agg = []
    for n in NAMES:
        d = load(k, n)
        p = d["pops"][:, 0] - d["t_launch"]
        tl_last = p[p <= BURN][-1] if (p <= BURN).any() else 0.0
        i1 = d["i0"] + int(round(tl_last / 0.01)) + 2
        ib = d["i0"] + int(round(BURN / 0.01))
        tg = d["TGT"][i1:ib]; tgo = d["TGO"][i1:ib] * d["STRETCH"][i1:ib]; left = BURN - d["tl"][i1:ib]
        reach = np.isfinite(tgo) & (tgo <= left)
        # min distance to the last chased target after the last pop until the end
        tgt = int(tg[tg >= 0][-1]) if (tg >= 0).any() else -1
        j = np.arange(i1, len(d["t"]) - 1); j = j[np.isfinite(d["R"][j, 0]) & (d["alt"][j] > 0)]
        dmin_t = np.min(np.linalg.norm(d["B"][j, tgt, :3] - d["R"][j, :3], axis=1)) if tgt >= 0 and j.size else np.nan
        # how many distinct targets chased in the tail
        agg.append([BURN - tl_last, reach.mean() if reach.size else np.nan, len(set(tg[tg >= 0].tolist())), dmin_t])
        print(f"  {n:15s} last pop t+{tl_last:5.2f}  tail {BURN - tl_last:4.2f}s  frac steps target reachable-before-burnout {agg[-1][1]:.2f}  "
              f"targets {agg[-1][2]}  min dist to last target {dmin_t:5.1f} m")
    A = np.array(agg)
    print(f"  tail mean {A[:,0].mean():.2f} s, reachable frac mean {np.nanmean(A[:,1]):.2f}, min-dist-to-last-target median {np.nanmedian(A[:,3]):.1f} m")

"""First hop (launch -> first pop): required pitch-over vs the hindsight direction and time to point within 20 / 10 deg;
closing speed toward the next balloon at each pop (state quality, context for the hop-model decomposition)."""
import json
import numpy as np
from common import *

A = json.load(open(OUT / "analyze_out.json"))
for k in ("their", "ours_strict", "ours_free"):
    F = [h for h in A[k]["hops"] if h["first"]]
    print(f"[{k}] first hop n={len(F)} duration med {np.median([h['hop'] for h in F]):.2f}s  turn at launch med {np.median([h['th0'] for h in F]):.1f} deg"
          f"  t20 med {np.nanmedian([h['t20'] for h in F]):.2f}s  t10 med {np.nanmedian([h['t10'] for h in F]):.2f}s  peak |w| med {np.median([h['wpk'] for h in F]):.2f}"
          f"  lost thrust (1-cos) med {np.median([h['lost'] for h in F]):.2f} m/s")
for k in ("their", "ours_strict", "ours_free"):
    C = []
    for n in NAMES:
        d = load(k, n)
        H = hops(d)
        for (ts, te, b, first), (ts2, te2, b2, _) in zip(H[:-1], H[1:]):
            i = idx(d["t"], te)
            r = d["B"][i, b2, :3] - d["p"][i]
            vr = d["v"][i] - d["B"][i, b2, 3:]
            C.append(vr @ unit(r))
    C = np.array(C)
    print(f"[{k}] closing speed toward the next balloon at the pop: median {np.median(C):.1f} m/s (p25 {np.percentile(C, 25):.1f}, p75 {np.percentile(C, 75):.1f}); negative in {np.mean(C < 0):.2f}")

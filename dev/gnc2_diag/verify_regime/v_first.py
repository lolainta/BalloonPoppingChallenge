"""First-pop check: was their first balloon on the ground at launch, how close were they at its release, and how much of the
summed strict first-pop gap do these runs carry.  Also the chance baseline: how many ground balloons lay within 5 m of their
rocket at the release moment (a skim over the field would pop *some* newly released balloon by luck)."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from vc import NAMES, get

ELEV = 20.0
tot = 0.0; pre = 0.0; rows = []
for n in NAMES:
    th, st = get("their", n), get("ours_strict", n)
    b = int(th["pops"][0, 1]); S = th["S"]; t = th["t"]
    rel = int(np.flatnonzero(S[:, b] >= 1)[0])
    on_ground_at_launch = S[th["i0"], b] == 0
    dist = np.linalg.norm(th["R"][rel, :3] - th["B"][rel, b, :3]) if np.isfinite(th["R"][rel, 0]) else np.nan
    ip = int(np.flatnonzero(S[:, b] == 2)[0])
    age = t[ip] - t[rel]
    alt = th["R"][ip, 2] - ELEV
    # ground balloons within 5 m (horizontal) of the rocket at the release step, and nearest ground spacing
    g = np.flatnonzero(S[rel - 1] == 0) if rel > 0 else np.array([], int)
    near = int((np.linalg.norm(th["B"][rel - 1, g, :2] - th["R"][rel, None, :2], axis=1) < 5).sum()) if np.isfinite(th["R"][rel, 0]) and g.size else -1
    gap = st["ps"][0] - th["ps"][0]
    tot += gap
    flag = np.isfinite(dist) and dist < 6
    if flag: pre += gap
    # strict: same balloon popped?  where was our rocket at release
    dS = np.linalg.norm(st["R"][rel, :3] - st["B"][rel, b, :3]) if rel < len(st["t"]) and np.isfinite(st["R"][rel, 0]) else np.nan
    print(f"{n:14s} seed {int(th['seed']):8d} b {b:2d} ground@launch {bool(on_ground_at_launch)!s:5s} rel-launch {t[rel]-th['tL']:5.2f} "
          f"their dist@rel {dist:6.1f} age@pop {age:5.2f} alt@pop {alt:5.1f} ground-balloons<5m {near:2d} n_ground {g.size:3d} | "
          f"strict first pop b {int(st['pops'][0,1]):2d} t+{st['ps'][0]:5.2f} (their t+{th['ps'][0]:5.2f}) gap {gap:5.2f} strict dist@rel {dS:5.1f} {'<-- pre' if flag else ''}")
print(f"summed first-pop gap {tot:.2f} s; runs with their dist@release < 6 m carry {pre:.2f} s ({pre/tot:.0%})")

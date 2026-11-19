"""Foreknowledge vs a seed-agnostic ground skim: for their runs, every balloon release while the rocket is below 8 m AGL
(first 8 s): distance rocket->balloon at release, whether it was popped within 1.5 s, and the approach rate over the
preceding 1 s (a rocket steering toward a still-grounded balloon before its release suggests knowing the schedule)."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from vc import NAMES, get

ELEV = 20.0
for n in NAMES:
    d = get("their", n); S, B, R, t = d["S"], d["B"], d["R"], d["t"]
    i0 = d["i0"]; out = []
    rel = {b: int(np.flatnonzero(S[:, b] >= 1)[0]) for b in range(100) if (S[:, b] >= 1).any()}
    low = 0; close = 0; popped_close = 0
    for b, r in sorted(rel.items(), key=lambda x: x[1]):
        if r <= i0 or t[r] - t[i0] > 8 or not np.isfinite(R[r, 0]): continue
        if R[r, 2] - ELEV > 8: continue
        low += 1
        dist = np.linalg.norm(R[r, :3] - B[r, b, :3])
        j = max(r - 100, i0 + 1)
        d1 = np.linalg.norm(R[j, :2] - B[r, b, :2])
        pop = (S[r:r + 150, b] == 2).any()
        # rank among grounded balloons by horizontal distance, 1 s before release
        g = np.flatnonzero(S[j] == 0)
        rank = int((np.linalg.norm(B[j, g, :2] - R[j, None, :2], axis=1) < d1).sum()) + 1
        if dist < 8: close += 1; popped_close += pop
        if dist < 12 or pop:
            out.append(f"b{b}@t+{t[r]-t[i0]:.2f} d{dist:.1f} (1s before {d1:.1f}, rank {rank}/{g.size}) {'POP' if pop else ''}")
    print(f"{n:14s} releases while rocket <8 m AGL: {low:2d}; within 8 m: {close}; popped: {popped_close} | " + "; ".join(out))

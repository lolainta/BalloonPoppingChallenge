"""Pops of young balloons (age < 5 s since release): where was the rocket when the balloon was released?
A rocket within a few metres of a ground balloon at the (unobservable, shuffled) release instant = seed-memorised timing.
Also time below 5 m altitude outside the launch phase.
.venv/bin/python dev/gnc2_diag/launch_regime/young.py
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import KINDS, NAMES, load  # noqa: E402

for k in KINDS:
    print(f"--- {k}")
    n_pre = 0
    low_after = []
    for n in NAMES:
        d = load(k, n)
        for j, (tp, b) in enumerate(d["pops"]):
            b = int(b); irel = int(np.argmax(d["S"][:, b] >= 1)); trel = d["t"][irel]
            if tp - trel < 5:
                i = irel
                dist = np.linalg.norm(d["R"][i, :3] - d["B"][i, b, :3]) if np.isfinite(d["R"][i, 0]) else np.nan
                # distance 1 s before release (committed approach?)
                i1 = max(irel - 100, 0)
                dist1 = np.linalg.norm(d["R"][i1, :3] - d["B"][i1, b, :3]) if np.isfinite(d["R"][i1, 0]) else np.nan
                pre = dist < 6
                n_pre += pre
                print(f"  {n:15s} pop#{j+1:2d} b{b:3d} t+{tp - d['t_launch']:5.2f}  age {tp - trel:4.2f}  dist@release {dist:6.1f}  dist@release-1s {dist1:6.1f}  alt@pop {d['alt'][int(np.argmin(np.abs(d['t'] - tp)))]:5.1f}{'  <-- pre-positioned' if pre else ''}")
        w = (d["tl"] > 6) & (d["tl"] <= 30)
        low_after.append((d["alt"][w] < 5).mean())
    print(f"  pre-positioned (<6 m at release): {n_pre};  frac of burn t+6..30 s below 5 m: median {np.median(low_after):.3f} mean {np.mean(low_after):.3f}")

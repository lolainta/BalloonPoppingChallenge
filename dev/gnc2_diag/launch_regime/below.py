"""Chasing from below: with thrust/weight 1.2-1.45 the net upward acceleration is only 2-4.4 m/s^2 while balloons rise at
~6 m/s. Fraction of burn time (t+1..30 s) the rocket is below the NEXT balloon it pops and climbing slower than it; mean
vertical gap; same definition (next popped balloon) for all three.
.venv/bin/python dev/gnc2_diag/launch_regime/below.py
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import KINDS, NAMES, load  # noqa: E402

for k in KINDS:
    fr, gap, frb = [], [], []
    for n in NAMES:
        d = load(k, n)
        p = d["pops"]; w = np.flatnonzero((d["tl"] >= 1) & (d["tl"] <= 30))
        nxt = np.searchsorted(p[:, 0], d["t"][w], side="left")
        ok = nxt < len(p); w, nxt = w[ok], nxt[ok]
        b = p[nxt, 1].astype(int)
        dz = d["B"][w, b, 2] - d["R"][w, 2]; dvz = d["B"][w, b, 5] - d["R"][w, 5]
        fr.append(np.mean((dz > 1.5) & (dvz > 0))); frb.append(np.mean(dz > 1.5)); gap.append(np.median(dz))
    print(f"{k:12s} below next balloon: {np.mean(frb):.2f} of burn;  below AND losing vertical ground: {np.mean(fr):.2f};  median vertical gap (balloon - rocket) {np.median(gap):5.1f} m")

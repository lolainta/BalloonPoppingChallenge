"""First-pop anatomy: when was their first balloon released relative to launch, where was each rocket at that moment,
what did ours_strict target before it was released, and balloon age at every pop (young / low balloons = ground regime).

.venv/bin/python dev/gnc2_diag/launch_regime/first_pop.py > dev/gnc2_diag/launch_regime/first_pop.txt
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import KINDS, NAMES, load  # noqa: E402

print("=== their first balloon: release time vs launch, rocket-balloon distance at release, time from release to pop ===")
print(f"{'name':15s} {'b':>3s} {'rel-launch':>10s} | {'their: dist@rel  pop-rel  alt@pop':>34s} | {'strict: dist@rel pop-rel alt@pop  targets before release (idx:s)':>60s}")
ages = {k: [] for k in KINDS}
agebins = {k: [] for k in KINDS}
for n in NAMES:
    th = load("their", n); st = load("ours_strict", n); fr = load("ours_free", n)
    b = int(th["pops"][0, 1])
    irel = int(np.argmax(th["S"][:, b] >= 1)); trel = th["t"][irel]
    s = f"{n:15s} {b:3d} {trel - th['t_launch']:10.2f} |"
    for d in (th, st):
        i = int(np.argmin(np.abs(d["t"] - trel)))
        dist = np.linalg.norm(d["R"][i, :3] - d["B"][i, b, :3])
        ip = np.flatnonzero(d["pops"][:, 1] == b)
        tp = d["pops"][ip[0], 0] if ip.size else np.nan
        s += f" {dist:8.1f} {tp - trel:8.2f} {d['alt'][int(np.argmin(np.abs(d['t'] - tp)))] if ip.size else np.nan:8.1f}"
        if "TGT" in d:
            w = (d["t"] >= d["t_launch"]) & (d["t"] <= max(trel, d["t_launch"] + 0.5))
            tg = d["TGT"][w]
            chg = [(int(tg[0]), 0.0)] + [(int(tg[j]), round(float(d["tl"][w][j]), 2)) for j in range(1, len(tg)) if tg[j] != tg[j - 1]]
            s += "  " + " ".join(f"{a}:{c}" for a, c in chg[:6])
        s += " |"
    print(s)

print("\n=== balloon age (t_pop - t_release) and rocket altitude at pop, all pops ===")
for k in KINDS:
    A, H = [], []
    for n in NAMES:
        d = load(k, n)
        for tp, b in d["pops"]:
            b = int(b)
            trel = d["t"][int(np.argmax(d["S"][:, b] >= 1))]
            A.append(tp - trel); H.append(d["alt"][int(np.argmin(np.abs(d["t"] - tp)))])
    A, H = np.array(A), np.array(H)
    print(f"{k:12s} n {len(A):3d}  age median {np.median(A):5.1f} s  age<3s {np.sum(A < 3):3d}  age<5s {np.sum(A < 5):3d}  "
          f"pop alt<10m {np.sum(H < 10):3d}  <30m {np.sum(H < 30):3d}  age quartiles {np.percentile(A, [25, 50, 75]).round(1)}")
    ages[k] = A

# first-pop split: close first balloon (dist at launch < 45 m) vs far
print("\n=== first-pop delay split by first-balloon distance at launch ===")
for thr_name, cond in (("close (<45 m)", lambda x: x < 45), ("far (>=45 m)", lambda x: x >= 45)):
    dts = {k: [] for k in KINDS}
    for n in NAMES:
        th = load("their", n)
        b = int(th["pops"][0, 1]); d0 = np.linalg.norm(th["B"][th["i0"], b, :3] - th["R"][th["i0"], :3])
        if not cond(d0):
            continue
        for k in KINDS:
            d = load(k, n)
            dts[k].append(d["pops"][0, 0] - d["t_launch"])
    print(thr_name, {k: (len(v), round(float(np.mean(v)), 2)) for k, v in dts.items()})

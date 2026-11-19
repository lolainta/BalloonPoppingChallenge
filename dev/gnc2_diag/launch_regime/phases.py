"""Where in the flight are pops lost?  Pop-time histogram since launch, active window (first..last pop), hop period, and the
end-of-burn state (vz, angle between body axis and velocity at burnout, post-burnout closest approaches).

.venv/bin/python dev/gnc2_diag/launch_regime/phases.py > dev/gnc2_diag/launch_regime/phases.txt
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import BURN, KINDS, NAMES, load  # noqa: E402

bins = np.array([0, 5, 10, 15, 20, 25, 28, 30, 32, 35, 60])
print("=== pops per time-since-launch bin (sum over 11 runs) ===")
print(f"{'bin (s)':12s}" + "".join(f"{f'{a}-{b}':>7s}" for a, b in zip(bins[:-1], bins[1:])) + "   total")
H = {}
for k in KINDS:
    tt = np.concatenate([load(k, n)["pops"][:, 0] - load(k, n)["t_launch"] for n in NAMES])
    H[k] = np.histogram(tt, bins)[0]
    print(f"{k:12s}" + "".join(f"{x:7d}" for x in H[k]) + f"   {len(tt):5d}")
print(f"{'their-strict':12s}" + "".join(f"{x:7d}" for x in H['their'] - H['ours_strict']))
print(f"{'their-free':12s}" + "".join(f"{x:7d}" for x in H['their'] - H['ours_free']))

print("\n=== window decomposition: pops = 1 + (last - first) / period ===")
for k in KINDS:
    F, L, N = [], [], []
    for n in NAMES:
        d = load(k, n); p = d["pops"][:, 0] - d["t_launch"]
        F.append(p[0]); L.append(p[-1]); N.append(len(p))
    F, L, N = map(np.array, (F, L, N))
    per = (L - F) / np.maximum(N - 1, 1)
    print(f"{k:12s} first {F.mean():5.2f} (med {np.median(F):5.2f})  last {L.mean():5.2f} (med {np.median(L):5.2f})  window {np.mean(L - F):5.2f}"
          f"  hops {np.mean(N - 1):5.2f}  period(window/hops) {np.sum(L - F) / np.sum(N - 1):5.2f}  pops {N.sum()}")

print("\n=== burnout state: vz, speed, alt, angle(body z, velocity), time since last pop; post-burnout min distance to any live balloon ===")
for k in KINDS:
    rows = []
    for n in NAMES:
        d = load(k, n)
        ib = d["i0"] + int(round(BURN / 0.01))
        R = d["R"][ib]; v = R[3:6]
        ang = np.degrees(np.arccos(np.clip(d["zb"][ib] @ v / np.linalg.norm(v), -1, 1)))
        p = d["pops"][:, 0] - d["t_launch"]
        before = p[p <= BURN]
        # after burnout: for each step until ground / end, min distance to balloons alive at that step
        j = np.arange(ib, len(d["t"]) - 1)
        j = j[np.isfinite(d["R"][j, 0]) & (d["alt"][j] > 0)]
        dmin = np.inf; n_close = 0
        if j.size:
            dist = np.linalg.norm(d["B"][j, :, :3] - d["R"][j, None, :3], axis=2)
            dist[d["S"][j] != 1] = np.inf
            md = dist.min(0)  # per balloon, min over time
            dmin = float(md[np.isfinite(md)].min()) if np.isfinite(md).any() else np.inf
            n_close = int(((md > 1.5) & (md < 4.0)).sum())
        rows.append([v[2], np.linalg.norm(v), d["alt"][ib], ang, BURN - (before[-1] if before.size else 0), (p > BURN).sum(),
                     dmin, n_close, (d["t"][j[-1]] - d["t"][ib]) if j.size else 0])
    A = np.array(rows, float)
    print(f"{k:12s} vz {np.median(A[:, 0]):6.1f}  speed {np.median(A[:, 1]):5.1f}  alt {np.median(A[:, 2]):6.1f}  body-vel angle med {np.median(A[:, 3]):5.1f} "
          f"(>{45}deg in {int((A[:, 3] > 45).sum())}/11)  last-pop->burnout med {np.median(A[:, 4]):4.2f} mean {A[:, 4].mean():4.2f}  "
          f"post-burn pops {int(A[:, 5].sum())}  near-misses 1.5-4 m {int(A[:, 7].sum())}  coast time med {np.median(A[:, 8]):4.1f}s")
    print("   per run angle:", np.round(A[:, 3]).astype(int).tolist(), " vz:", np.round(A[:, 0], 1).tolist())

print("\n=== their post-burnout pops: time after burnout, rocket alt, balloon age ===")
for n in NAMES:
    d = load("their", n)
    for tp, b in d["pops"]:
        if tp - d["t_launch"] > BURN:
            i = int(np.argmin(np.abs(d["t"] - tp)))
            print(f"  {n:15s} +{tp - d['t_launch'] - BURN:5.2f}s after burnout  alt {d['alt'][i]:6.1f}  speed {np.linalg.norm(d['R'][i, 3:6]):5.1f}")

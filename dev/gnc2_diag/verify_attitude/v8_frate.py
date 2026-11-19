"""V8: angular rate of the wanted direction F in the last 0.5 s before a pop (not within 1 s of a target switch)."""
import numpy as np
from base import *
FR = []
for n in NAMES:
    d = get("ours_strict", n); t = d["t"]; F = d["F"]; tgt = d["TGT"]
    g = d["burnmask"] & np.isfinite(F).all(1); g[:d["i0"] + 50] = False
    uF = np.nan_to_num(F) / np.maximum(np.linalg.norm(np.nan_to_num(F), axis=1, keepdims=True), 1e-9)
    rate = np.r_[np.nan, angd(uF[1:], uF[:-1]) / DT]
    sw = np.flatnonzero((tgt[1:] != tgt[:-1]) & (tgt[1:] >= 0)) + 1
    recent = np.zeros(len(t), bool)
    for s in sw: recent[s:s + 101] = True
    last = np.zeros(len(t), bool)
    for tp, _ in d["pops"]:
        k = ti(t, tp); last[max(0, k - 50):k + 1] = True
    m = g & np.r_[False, g[:-1]] & last & ~recent
    FR.append(rate[m])
FR = np.concatenate(FR)
print(f"F rate last 0.5 s: median {np.median(FR):.0f} deg/s, frac > 143 deg/s (2.5 rad/s) {np.mean(FR > np.degrees(2.5)):.3f}, n {len(FR)}")

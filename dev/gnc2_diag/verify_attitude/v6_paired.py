"""V6: paired variant comparison from dev/gnc2_diag/attitude/runs (read-only), deduplicated (9 unique strict flights).
Metrics per replay: pops in burn; (t_n - t_1)/(n-1) over common n (their metric); mean hop incl. first over the burn
(= (t_last - L)/n_burn); time of the k-th pop averaged over common k; |w| at pops."""
import numpy as np
from base import *

RUNS = ROOT / "dev/gnc2_diag/attitude/runs"
U = [n for n in NAMES if n not in ("13_Apex", "18_Taoyuan")]


def st(v, n):
    f = RUNS / v / f"{n}.npz"
    if not f.exists(): return None
    d = dict(np.load(f)); t, R = d["t"], d["R"]
    live = ~np.isnan(R[:, 0]); L = t[np.argmax(live)]
    pt = np.sort(d["pops"][:, 0]) - L if len(d["pops"]) else np.array([])
    pb = pt[pt <= BURN]
    wp = [np.linalg.norm(R[int(round((x + L - t[0]) / DT)), 10:12]) for x in pb]
    return dict(pb=pb, n=len(pb), wpop=np.array(wp))


base = {n: st("base", n) for n in U}
se = lambda x: np.std(x, ddof=1) / np.sqrt(len(x))
print(f"{'variant':<15}{'pops(9)':>8}{'dpops':>12}{'d (tn-t1)/(n-1)':>20}{'d mean kth-pop time':>22}{'|w|pop med':>11}")
for v in ["base", "katt2_mr1.5", "katt4_mr3", "mr4", "katt5_mr4_kr15", "sqrt", "sqrt_c", "box", "ff", "sqrt_box_ff", "hold03", "hold05", "fast_physics", "fast_physics8"]:
    S = {n: st(v, n) for n in U}
    if any(s is None for s in S.values()): continue
    dp = np.array([S[n]["n"] - base[n]["n"] for n in U])
    dh, dk = [], []
    for n in U:
        a, b = base[n]["pb"], S[n]["pb"]; k = min(len(a), len(b))
        if k >= 2: dh.append(((b[k-1] - b[0]) - (a[k-1] - a[0])) / (k - 1))
        if k >= 1: dk.append(np.mean(b[:k] - a[:k]))
    dh, dk = np.array(dh), np.array(dk)
    wp = np.concatenate([S[n]["wpop"] for n in U])
    print(f"{v:<15}{sum(S[n]['n'] for n in U):>8d}{dp.mean():+7.2f}±{se(dp):.2f}{dh.mean():+12.3f}±{se(dh):.3f}{dk.mean():+14.2f}±{se(dk):.2f}{np.median(wp):>11.2f}")

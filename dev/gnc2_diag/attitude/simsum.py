"""Summarize the strict-order attitude variants (runs/<variant>/<name>.npz) against runs/base.
Per variant: total pops, pops in the burn, median/mean hop (not the first), launch->first pop, paired shift of the k-th
pop time vs base (common k), body rate p95/max, logged gimbal saturation, nose-vs-F angle, crash count."""
import sys
import numpy as np
from common import *

RUNS = OUT / "runs"


def stats(v, n):
    f = RUNS / v / f"{n}.npz"
    if not f.exists():
        return None
    d = dict(np.load(f))
    t, R = d["t"], d["R"]
    live = ~np.isnan(R[:, 0]); i0 = int(np.argmax(live)); L = t[i0]
    burn = live & (t >= L + 0.5) & (t <= L + BURN - 0.02)
    pops = d["pops"]; pt = np.sort(pops[:, 0]) if len(pops) else np.array([])
    pb = pt[pt <= L + BURN]
    hop = np.diff(np.r_[L, pb])
    zb = quat_to_R(np.where(live[:, None], R[:, 6:10], [1, 0, 0, 0]))[..., :, 2]
    F = d["F"]; g = np.isfinite(F).all(1) & burn
    th = np.degrees(ang(zb[g], F[g]))
    wxy = np.linalg.norm(R[burn, 10:12], axis=1)
    gim = np.hypot(d["A"][burn, 0], d["A"][burn, 1])
    crash = live[-1] if False else bool((R[live, 2] - R[i0, 2] < 0.5).any() and t[live][-1] < L + BURN)
    return dict(pops=len(pt), pb=len(pb), hops=hop[1:], first=hop[0] if len(hop) else np.nan, pt=pb - L,
                w95=np.percentile(wxy, 95), wmax=wxy.max(), sat=(gim >= 14.5).mean(), th_med=np.median(th) if len(th) else np.nan,
                th20=(th > 20).mean() if len(th) else np.nan, dur=t[live][-1] - L)


def main(variants):
    base = {n: stats("base", n) for n in NAMES}
    print(f"{'variant':<16}{'pops':>5}{'burn':>5}{'hop med':>8}{'hop mean':>9}{'first':>7}{'dk-pop':>8}{'w p95':>7}{'w max':>7}{'sat':>6}"
          f"{'thF med':>8}{'thF>20':>7}{'short':>6}")
    for v in variants:
        S = {n: stats(v, n) for n in NAMES}
        S = {n: s for n, s in S.items() if s is not None}
        if not S:
            continue
        hops = np.concatenate([s["hops"] for s in S.values()])
        dk = []
        for n, s in S.items():
            b = base[n]; k = min(len(b["pt"]), len(s["pt"]))
            dk += list(s["pt"][:k] - b["pt"][:k])
        short = sum(s["dur"] < BURN - 1 for s in S.values())
        m = lambda key: np.nanmean([s[key] for s in S.values()])
        print(f"{v:<16}{sum(s['pops'] for s in S.values()):5d}{sum(s['pb'] for s in S.values()):5d}"
              f"{np.median(hops) if len(hops) else np.nan:8.2f}{np.mean(hops) if len(hops) else np.nan:9.2f}{np.nanmedian([s['first'] for s in S.values()]):7.2f}"
              f"{np.mean(dk) if dk else np.nan:8.2f}{m('w95'):7.2f}{np.max([s['wmax'] for s in S.values()]):7.2f}{m('sat'):6.3f}"
              f"{m('th_med'):8.1f}{m('th20'):7.3f}{short:4d}/{len(S)}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["base", "katt2_mr1.5", "katt4_mr3", "mr4", "katt5_mr4_kr15", "sqrt", "sqrt_box_ff", "fast_physics", "fast_physics8"])

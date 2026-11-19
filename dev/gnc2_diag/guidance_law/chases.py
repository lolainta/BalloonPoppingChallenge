"""Chase-level table for their / ours_strict / ours_free (data/gnc2). One row per chase:
run, name, k, balloon, t0, t1 (pop or end), popped, dur, T1 (min reach time at chase start, constant-direction full thrust,
coast-aware, balloon constant velocity, same a_thrust profile for both), TGO0 (our GNC's own logged value), stretch
(mean over chase, first value), passes (distance local minima before the end, d<40 m), miss passes (d > 1.5 m), first miss
distance, time from first miss pass to end, rel speed at hit, closing speed at hit.
.venv/bin/python dev/gnc2_diag/guidance_law/chases.py  -> dev/gnc2_diag/guidance_law/chases.npz (+ printed summary)
"""
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
D = ROOT / "data/gnc2"
OUT = Path(__file__).resolve().parent
G = np.array([0.0, 0.0, -9.81])
NAMES = "10_CoolBa 10_SpaceY 11_Mochi 11_near_miss 12_Apex 13_Apex 13_Yaka 14_SpaceBerry 16_SpaceBerry 16_TeamDJ 18_Taoyuan".split()
GRID = np.arange(0.0, 15.0, 0.02)
TOL = 1.5
BURN = 30.0
HANDOFF = 1.5  # (s) a pop this long after switching away still counts for the chase


def reach(p, v, bp, bv, a, tb, tol=None):
    """Min time to hit: |r + vr t - G t^2/2| <= a * disp(t) + tol, disp = t^2/2 while burning, then coast.
    tol = TOL (1.5 m, the balloon radius = the real hit criterion; our GNC plans with 0)."""
    tol = TOL if tol is None else tol
    t = GRID[:, None]
    miss = (bp - p)[None] + (bv - v)[None] * t - 0.5 * G[None] * t ** 2
    disp = np.where(GRID <= tb, 0.5 * GRID ** 2, tb * (GRID - 0.5 * tb))
    ok = np.linalg.norm(miss, axis=1) <= a * disp + tol
    return GRID[np.argmax(ok)] if ok.any() else np.inf


def athrust_profile():
    """Full-throttle thrust accel vs time since launch: median of our logged estimates over the strict runs."""
    tt = np.arange(0, 30.01, 0.1); acc = []
    for n in NAMES:
        u = np.load(D / f"ours_strict_{n}.npz"); L = float(u["launch"]); m = (u["t"] > L + 0.3) & (u["t"] < L + 30)
        acc.append(np.interp(tt, u["t"][m] - L, u["a_thrust"][m]))
    return tt, np.median(acc, 0)


TT, AT = athrust_profile()


def a_at(tl):
    return float(np.interp(tl, TT, AT))


def passes(t, d, i0, i1):
    """Local minima of d on [i0, i1) (excluding i1 itself)."""
    seg = d[i0:i1 + 1]
    if len(seg) < 3:
        return []
    k = np.flatnonzero((seg[1:-1] < seg[:-2]) & (seg[1:-1] <= seg[2:])) + 1
    return [(t[i0 + j], seg[j]) for j in k if seg[j] < 40.0 and i0 + j < i1 - 1]


def chases_of(kind, n):
    f = np.load(D / f"{kind}_{n}.npz")
    t, R, B, S = f["t"], f["R"], f["B"].astype(float), f["S"]
    L = float(f["launch"]); pops = f["pops"]
    ptime = {int(b): tp for tp, b in pops}
    rows = []
    if kind == "their":
        segs = []
        prev = L
        for tp, b in pops:
            i0 = int(np.searchsorted(t, prev - 1e-6)); i1 = int(np.searchsorted(t, tp - 1e-6))
            segs.append((i0, i1, int(b), True, 0.0)); prev = tp
    else:
        TGT = f["TGT"]; segs = []
        i = 0; T = len(t)
        while i < T:
            if TGT[i] < 0:
                i += 1; continue
            j = i
            while j + 1 < T and TGT[j + 1] == TGT[i]:
                j += 1
            b = int(TGT[i]); tp = ptime.get(b)
            # popped: during the chase, or within HANDOFF s after the policy already switched away (coasting through;
            # free mode switches ~0.5 s early) as long as it was not re-targeted before the pop
            later = np.flatnonzero(TGT[j + 1:] == b)
            nxt = t[j + 1 + later[0]] if later.size else np.inf
            popped = tp is not None and t[i] - 0.02 <= tp <= min(t[j] + HANDOFF, nxt - 0.005)
            i1 = int(np.searchsorted(t, tp - 1e-6)) if popped else j
            segs.append((i, i1, b, popped, (tp - t[j]) if popped else np.nan)); i = j + 1
    for k, (i0, i1, b, popped, lead) in enumerate(segs):
        p0, v0 = R[i0, :3], R[i0, 3:6]
        if np.isnan(p0).any():  # pre-launch chase start (their first): use the pad state at launch
            il = int(np.argmax(~np.isnan(R[:, 0]))); p0, v0 = R[il, :3], R[il, 3:6]
        tl = max(t[i0] - L, 0.0)
        T1 = reach(p0, v0, B[i0, b, :3], B[i0, b, 3:], a_at(tl), max(BURN - tl, 0.01))
        d = np.linalg.norm(B[:, b, :3] - R[:, :3], axis=1)
        ps = passes(t, d, i0, i1)
        miss = [(tp_, dm) for tp_, dm in ps if dm > 1.5]
        vr = R[i1, 3:6] - B[i1, b, 3:]
        rhat = (B[i1, b, :3] - R[i1, :3]); rhat = rhat / max(np.linalg.norm(rhat), 1e-9)
        row = dict(kind=kind, name=n, k=k, b=b, t0=t[i0], t1=t[i1], tl0=tl, popped=popped, dur=t[i1] - t[i0], T1=T1,
                   d0=d[i0], npass=len(ps), nmiss=len(miss), miss1=miss[0][1] if miss else np.nan,
                   lost=(t[i1] - miss[0][0]) if (miss and popped) else np.nan, vrel=np.linalg.norm(vr), vclose=vr @ rhat,
                   speed=np.linalg.norm(R[i1, 3:6]), lead=lead, tend=t[min(i1, len(t) - 1)] if kind == 'their' else None)
        if kind != "their":
            st = f["STRETCH"][i0:i1 + 1]; tg = f["TGO"][i0:i1 + 1]
            row.update(s_mean=float(st.mean()), s0=float(st[0]), TGO0=float(tg[0]),
                       s_frac=[float((np.isclose(st, s)).mean()) for s in (0.6, 0.8, 1.0, 1.3)])
        rows.append(row)
    return rows


if __name__ == "__main__":
    import pickle
    allrows = []
    for n in NAMES:
        for kind in ("their", "ours_strict", "ours_free"):
            allrows += chases_of(kind, n)
    pickle.dump(allrows, open(OUT / "chases.pkl", "wb"))
    print(len(allrows), "rows")

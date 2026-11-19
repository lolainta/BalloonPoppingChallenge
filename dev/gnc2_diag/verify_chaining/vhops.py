"""Independent recomputation of the hop-geometry claims (verify_chaining). Pops are taken from the status array S
(first step with status 2); the pop state is the step where the status changes (index i) -- NOT i-1 -- as a robustness
check of the time alignment. Specific force from a central difference of the logged velocity; drag from the rocket body
parameters with the *current* mass unknown -> bounded using dry mass (upper bound). Writes vhops_<kind>.csv."""
import os, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[3]
D = Path(os.environ.get("VH_DIR", ROOT / "data/gnc2"))
NAMES = "10_CoolBa 10_SpaceY 11_Mochi 11_near_miss 12_Apex 13_Apex 13_Yaka 14_SpaceBerry 16_SpaceBerry 16_TeamDJ 18_Taoyuan".split()
g = np.array([0, 0, -9.81]); DT = 0.01; BURN = 30.0
CD, RAD, MDRY = 0.25, 0.22, 73.3
KD = 0.5 * CD * np.pi * RAD ** 2 / MDRY


def rotm(q):  # scalar-first unit quaternion, body -> world
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def angd(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9: return np.nan
    return float(np.degrees(np.arccos(np.clip(a @ b / na / nb, -1, 1))))


def spec_force(R):
    v = R[:, 3:6]; a = np.full_like(v, np.nan)
    a[1:-1] = (v[2:] - v[:-2]) / (2 * DT)
    rho = 1.225 * np.exp(-R[:, 2:3] / 8500.0)
    drag = -KD * rho * np.linalg.norm(v, axis=1, keepdims=True) * v
    return a - g - drag


def tmin(p, v, bp, bv, a, rhit=1.5, tmax=15.0):
    """smallest t: |bp-p+(bv-v)t - g t^2/2| - a t^2/2 - rhit <= 0 (continuous, fine scan + bisection)."""
    def h(t):
        m = bp - p + (bv - v) * t - 0.5 * g * t * t
        return np.linalg.norm(m) - 0.5 * a * t * t - rhit
    if h(0.0) <= 0: return 0.0
    ts = np.arange(0.0, tmax, 0.005)
    m = (bp - p)[None] + (bv - v)[None] * ts[:, None] - 0.5 * g[None] * ts[:, None] ** 2
    hv = np.linalg.norm(m, axis=1) - 0.5 * a * ts ** 2 - rhit
    k = np.flatnonzero(hv <= 0)
    if len(k) == 0: return np.inf
    lo, hi = ts[k[0] - 1], ts[k[0]]
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if h(mid) <= 0: hi = mid
        else: lo = mid
    return hi


def flight(kind, name):
    z = np.load(D / f"{kind}_{name}.npz")
    t, R, B, S = z["t"], z["R"], z["B"].astype(float), z["S"]
    live = ~np.isnan(R[:, 0]); il = int(np.argmax(live)); launch = t[il]
    burn = float(z["burn_time"]) if "burn_time" in z.files else BURN
    pops = []
    for b in range(S.shape[1]):
        c = np.flatnonzero(S[:, b] == 2)
        if len(c): pops.append((int(c[0]), b))
    pops.sort()
    f = spec_force(R); fm = np.linalg.norm(f, axis=1)
    inb = live & (t > launch + 0.5) & (t < launch + burn - 0.5)
    cap = float(np.nanpercentile(fm[inb], 95))
    # nose vs specific force check (convention)
    ang_nf = [angd(rotm(R[i, 6:10])[:, 2], f[i]) for i in np.flatnonzero(inb)[::10]]
    P = [(i, b) for i, b in pops if t[i] <= launch + burn]
    rows = []
    for k in range(len(P) - 1):
        (i0, b0), (i1, b1) = P[k], P[k + 1]
        p, v = R[i0, :3], R[i0, 3:6]
        bp, bv = B[i0, b1, :3], B[i0, b1, 3:6]
        r = bp - p; d = np.linalg.norm(r); rh = r / d; vr = v - bv
        vc = vr @ rh; vp = np.linalg.norm(vr - vc * rh)
        seg = slice(i0, i1)
        sp = np.linalg.norm(R[seg, 3:6], axis=1)
        tm = tmin(p, v, bp, bv, cap); tma = tmin(p, bv + np.linalg.norm(vr) * rh, bp, bv, cap)
        # min-time thrust direction at the pop (for nose error)
        mm = bp - p + (bv - v) * tm - 0.5 * g * tm * tm
        fdir = mm / np.linalg.norm(mm)
        fs = f[seg]; ok = np.all(np.isfinite(fs), 1)
        cosf = (fs @ fdir) / np.maximum(np.linalg.norm(fs, axis=1), 1e-9)
        w = np.flatnonzero(ok & (cosf > np.cos(np.radians(15))))
        rows.append(dict(kind=kind, name=name, k=k, b0=b0, b1=b1, t0=t[i0], T=t[i1] - t[i0], d=d, th=angd(vr, r), th_ground=angd(v, r),
                         vclose=vc, vperp=vp, speed=np.linalg.norm(v), vrel_hit=np.linalg.norm(v - B[i0, b0, 3:6]),
                         vmin_ratio=sp.min() / np.linalg.norm(v), vturn=angd(v, R[i1, 3:6]),
                         dv_int=float(np.nansum(fm[seg]) * DT), fuse=float(np.nanmean(fm[seg]) / cap), cap=cap,
                         Tmin=tm, Tmin_al=tma, nose=angd(rotm(R[i0, 6:10])[:, 2], fdir),
                         t_turn=(w[0] * DT if len(w) else np.nan), el_v=np.degrees(np.arcsin(v[2] / np.linalg.norm(v))),
                         dz=r[2], alt=p[2], dh=np.linalg.norm((R[i1, :2] - p[:2])), dzp=R[i1, 2] - p[2],
                         px=p[0], py=p[1], pz=p[2], rx=r[0], ry=r[1], rz=r[2], vx=v[0], vy=v[1], vz=v[2],
                         st_next=int(S[i0, b1])))
    return rows, dict(kind=kind, name=name, launch=launch, npops=len(pops), npops_burn=len(P), cap=cap,
                      nose_force_med=np.nanmedian(ang_nf), nose_force_p95=np.nanpercentile(ang_nf, 95),
                      first=(t[P[0][0]] - launch) if P else np.nan, popseq=tuple(b for _, b in pops), poptimes=tuple(np.round(t[[i for i, _ in pops]], 2)))


if __name__ == "__main__":
    here = Path(__file__).parent
    kinds = sys.argv[1:] or ["their", "ours_strict", "ours_free"]
    F = []
    for kind in kinds:
        rows = []
        for n in NAMES:
            if (D / f"{kind}_{n}.npz").exists():
                r, info = flight(kind, n); rows += r; F.append(info)
        pd.DataFrame(rows).to_csv(here / f"vhops_{kind}{os.environ.get('VH_TAG','')}.csv", index=False)
    pd.DataFrame(F).to_csv(here / f"vflights{os.environ.get('VH_TAG','')}.csv", index=False)
    print(pd.DataFrame(F).drop(columns=["popseq", "poptimes"]).round(2).to_string())

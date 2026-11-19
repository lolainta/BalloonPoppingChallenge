"""Per-hop geometry at each pop: their / ours_strict / ours_free (data/gnc2).

For every consecutive pair of pops (k -> k+1) inside the burn:
  d        distance rocket -> next balloon at pop k (m)
  th_los   angle between rocket velocity and line of sight to the next balloon (deg)
  th_int   angle between rocket-velocity-relative-to-next-balloon and the LOS (deg) (the relevant one: balloon moves ~7 m/s)
  vclose   closing speed on the next balloon at pop k = -(v - vb).r_hat (m/s)
  vperp    |component of relative velocity perpendicular to LOS| (m/s)
  speed    |v| at pop k; vrel_hit relative speed to the popped balloon at pop k
  T        hop time t_{k+1} - t_k (s)
  dv_int   integral of |thrust accel| over the hop (m/s) (thrust accel = dv/dt - g + drag)
  dv_net   |v(t_{k+1}) - v(t_k) - g T| (m/s) (minimum impulse that the hop needed, same endpoints)
  nose     angle between nose axis at pop k and the direction the min-time constant-force solution needs (deg)
  Tmin     min reach time of the next balloon from the state at pop k (constant force |f| <= a_thr, gravity, balloon
           constant velocity; same model as GNCAgent._reach_time with margin 0, no coast)
  Tmin_al  same with the velocity rotated onto the LOS-to-intercept direction, same speed (ideal chaining)
  Tmin_0   same with zero velocity relative to... (rocket at rest)
Writes hops_<kind>.csv and prints summaries.
"""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
import os
D = Path(os.environ["HOPS_DIR"]) if os.environ.get("HOPS_DIR") else ROOT / "data/gnc2"
NAMES = "10_CoolBa 10_SpaceY 11_Mochi 11_near_miss 12_Apex 13_Apex 13_Yaka 14_SpaceBerry 16_SpaceBerry 16_TeamDJ 18_Taoyuan".split()
G = np.array([0.0, 0.0, -9.81])
K_DRAG = 2.5937e-4
BURN = 30.0
GRID = np.arange(0.01, 15.0, 0.01)


def quat_R(q):
    e0, e1, e2, e3 = q
    return np.array([[1 - 2 * (e2 * e2 + e3 * e3), 2 * (e1 * e2 - e0 * e3), 2 * (e1 * e3 + e0 * e2)],
                     [2 * (e1 * e2 + e0 * e3), 1 - 2 * (e1 * e1 + e3 * e3), 2 * (e2 * e3 - e0 * e1)],
                     [2 * (e1 * e3 - e0 * e2), 2 * (e2 * e3 + e0 * e1), 1 - 2 * (e1 * e1 + e2 * e2)]])


def thrust_acc(t, R):
    """thrust specific force (world) from the state log: dv/dt - g + drag."""
    v = R[:, 3:6]
    a = np.gradient(v, t, axis=0)
    rho = 1.225 * np.clip(1.0 - 2.2558e-5 * R[:, 2:3], 0.5, 1.0) ** 4.2559
    drag = -K_DRAG * rho * np.linalg.norm(v, axis=1, keepdims=True) * v  # ~no wind term (scenario 1 wind small)
    return a - G[None] - drag


R_HIT = 1.5  # pop = rocket centre within the balloon radius (balloon_world._detect_pops)


def tmin(p, v, bp, bv, a, r_hit=R_HIT):
    """min t with |bp - p + (bv - v) t - G t^2/2| <= a t^2/2 + r_hit -> (t, unit force direction at that solution)."""
    tg = GRID[:, None]
    miss = (bp - p)[None] + (bv - v)[None] * tg - 0.5 * G[None] * tg ** 2
    ok = np.linalg.norm(miss, axis=1) <= a * 0.5 * GRID ** 2 + r_hit
    if not ok.any():
        return np.inf, np.full(3, np.nan)
    i = int(np.argmax(ok))
    return GRID[i], miss[i] / max(np.linalg.norm(miss[i]), 1e-9)


def hop_force(fa, fmag, j0, i1, a_cap, fdir):
    """thrust use during the hop: mean |f| / capacity, direction consistency |sum f| / sum |f|, time until the thrust
    points within 15 deg of the min-time direction fdir (turn), share of impulse along fdir."""
    f = fa[j0:max(i1 - 1, j0 + 1)]; m = fmag[j0:max(i1 - 1, j0 + 1)]
    ok = np.isfinite(m)
    f, m = f[ok], m[ok]
    if len(m) == 0:
        return dict(thr_use=np.nan, consist=np.nan, t_turn=np.nan, along=np.nan)
    cosf = (f @ fdir) / np.maximum(m, 1e-6) if np.all(np.isfinite(fdir)) else np.full(len(m), np.nan)
    within = np.flatnonzero(cosf > np.cos(np.radians(15)))
    return dict(thr_use=float(m.mean() / a_cap), consist=float(np.linalg.norm(f.sum(0)) / max(m.sum(), 1e-6)),
                t_turn=float(within[0] * 0.01) if len(within) else np.nan, along=float((f @ fdir).sum() / max(m.sum(), 1e-6)))


def ang(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-6 or nb < 1e-6:
        return np.nan
    return np.degrees(np.arccos(np.clip(a @ b / (na * nb), -1, 1)))


def hops(kind, name):
    z = np.load(D / f"{kind}_{name}.npz")
    t, R, B, pops, S = z["t"], z["R"], z["B"], z["pops"], z["S"]
    launch = float(z["launch"])
    live = ~np.isnan(R[:, 0])
    fa = np.full_like(R[:, :3], np.nan)
    idx = np.flatnonzero(live)
    fa[idx] = thrust_acc(t[idx], R[idx])
    fmag = np.linalg.norm(fa, axis=1)
    # thrust capability: the 95th percentile of |f| during the burn (full throttle), per run
    burn = live & (t > launch + 0.5) & (t < launch + BURN - 0.5)
    a_cap = float(np.nanpercentile(fmag[burn], 95))
    rows = []
    P = [(float(tp), int(b)) for tp, b in pops if tp <= launch + BURN]
    seq = [(launch, -1)] + P
    for k in range(len(seq) - 1):
        (t0, b0), (t1, b1) = seq[k], seq[k + 1]
        i0 = int(np.argmin(np.abs(t - t0))); i1 = int(np.argmin(np.abs(t - t1)))
        if k == 0:
            i0 = int(np.argmax(live))
        # state at pop k: the last step before the status change (index i0 - 1 has status 1)
        j0 = max(i0 - 1, 0) if k > 0 else i0
        p, v = R[j0, :3], R[j0, 3:6]
        bp, bv = B[j0, b1, :3].astype(float), B[j0, b1, 3:6].astype(float)
        r = bp - p; d = np.linalg.norm(r); rh = r / d
        vr = v - bv
        vclose = vr @ rh
        vperp = np.linalg.norm(vr - vclose * rh)
        T = t1 - t0
        seg = slice(j0, max(i1 - 1, j0 + 1))
        dv_int = float(np.nansum(fmag[seg]) * 0.01)
        dv_net = float(np.linalg.norm(R[max(i1 - 1, 0), 3:6] - v - G * T))
        tm, fdir = tmin(p, v, bp, bv, a_cap)
        # ideal chaining: same speed relative to the balloon, pointed along the intercept LOS
        tm_al, _ = tmin(p, bv + np.linalg.norm(vr) * rh, bp, bv, a_cap)
        tm_0, _ = tmin(p, np.zeros(3), bp, bv, a_cap)
        tm_b, _ = tmin(p, bv, bp, bv, a_cap)  # co-moving with the balloon (zero relative velocity)
        tm_c, _ = tmin(p, v, bp, bv, a_cap, 0.0)  # centre hit
        nose = quat_R(R[j0, 6:10])[:, 2]
        # hop path: min speed, speed at the next pop, velocity turn, horizontal / vertical displacement
        sp = np.linalg.norm(R[j0:max(i1 - 1, j0 + 1), 3:6], axis=1)
        v1 = R[max(i1 - 1, 0), 3:6]; p1 = R[max(i1 - 1, 0), :3]
        rows.append(dict(name=name, k=k, b0=b0, b1=b1, t0=t0, T=T, d=d, th_los=ang(v, r), th_int=ang(vr, r),
                         vclose=vclose, vperp=vperp, speed=np.linalg.norm(v), vrel_nb=np.linalg.norm(vr),
                         vrel_hit=(np.linalg.norm(v - B[j0, b0, 3:6]) if b0 >= 0 else np.nan),
                         vz=v[2], alt=p[2] - 20.0, dz=r[2], dv_int=dv_int, dv_net=dv_net,
                         nose_err=ang(nose, fdir), vel_nose=ang(nose, v), Tmin=tm, Tmin_al=tm_al, Tmin_0=tm_0, Tmin_b=tm_b,
                         a_cap=a_cap, tb_left=launch + BURN - t0, Tmin_c=tm_c, st_b1=int(S[j0, b1]),
                         el_los=np.degrees(np.arcsin(rh[2])), el_v=np.degrees(np.arcsin(v[2] / max(np.linalg.norm(v), 1e-6))),
                         vmin_hop=sp.min(), speed1=np.linalg.norm(v1), vturn=ang(v, v1),
                         dh=np.linalg.norm((p1 - p)[:2]), dzp=(p1 - p)[2], dv_kin=np.linalg.norm(v1 - v),
                         px=p[0], py=p[1], pz=p[2], vx=v[0], vy=v[1], vzz=v[2], rx=r[0], ry=r[1], rz=r[2],
                         fdx=fdir[0], fdy=fdir[1], fdz=fdir[2], **hop_force(fa, fmag, j0, i1, a_cap, fdir)))
    return rows


def load(kind):
    out = []
    for n in NAMES:
        if (D / f"{kind}_{n}.npz").exists():
            out += hops(kind, n)
    return out


if __name__ == "__main__":
    import csv
    here = Path(__file__).parent
    kinds = sys.argv[1:] or ("their", "ours_strict", "ours_free")
    for kind in kinds:
        rows = load(kind)
        with open(here / f"hops_{kind}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(kind, len(rows), "hops (incl. launch->first)")

"""Independent re-derivation helpers (verifier). Axial specific force from the logged quaternion; drag along -body z
(RocketPy R3 = -0.5 rho V^2 A Cd along body axis), thrust model fitted from our own throttle==1 samples."""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DT = 0.01
G = np.array([0.0, 0.0, -9.81])
NAMES = "10_CoolBa 10_SpaceY 11_Mochi 11_near_miss 12_Apex 13_Apex 13_Yaka 14_SpaceBerry 16_SpaceBerry 16_TeamDJ 18_Taoyuan".split()
AREA = np.pi * 0.22 ** 2
CD = 0.25
BURN = 30.0


def load(kind, name, sims=False):
    p = ROOT / (f"dev/gnc2_diag/thrust_util/sims/{kind}_{name}.npz" if sims else f"data/gnc2/{kind}_{name}.npz")
    d = np.load(p)
    return {k: d[k] for k in d.files}


def nose_of(q):
    """body z axis in world for scalar-first body->world quaternion (rotation matrix third column)."""
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    e0, e1, e2, e3 = np.moveaxis(q, -1, 0)
    return np.stack([2 * (e1 * e3 + e0 * e2), 2 * (e2 * e3 - e0 * e1), e0**2 - e1**2 - e2**2 + e3**2], -1)


def rho_of(z):
    return 1.225 * np.exp(-np.maximum(z, 0.0) / 8500.0)


def derive(d, m_of=None):
    """Forward differences: interval k = (t[k-1], t[k]] is driven by action A[k] (logged after env.step(A[k]))."""
    t, R = d["t"], d["R"]
    L = float(d["launch"])
    v = R[:, 3:6]
    a = np.full_like(v, np.nan)
    a[1:] = (v[1:] - v[:-1]) / DT
    f_sp = a - G
    nz = nose_of(R[:, 6:10])
    nmid = np.full_like(nz, np.nan)
    nmid[1:] = nz[1:] + nz[:-1]
    nmid /= np.linalg.norm(nmid, axis=1, keepdims=True)
    vmid = np.full_like(v, np.nan); vmid[1:] = 0.5 * (v[1:] + v[:-1])
    zmid = np.full(len(t), np.nan); zmid[1:] = 0.5 * (R[1:, 2] + R[:-1, 2])
    tau = t - L  # end of interval k, relative to launch
    taum = tau - 0.5 * DT
    m = m_of(taum) if m_of is not None else 88.0 - 0.587 * np.clip(taum, 0, BURN)
    D = 0.5 * rho_of(zmid) * np.sum(vmid**2, 1) * AREA * CD / m  # drag accel magnitude along -nose
    f_ax = np.einsum("ij,ij->i", f_sp, nmid)  # axial specific force
    f_perp = np.linalg.norm(f_sp - f_ax[:, None] * nmid, axis=1)
    thr_ax = f_ax + D  # thrust accel along the nose
    return dict(t=t, tau=tau, taum=taum, L=L, p=R[:, :3], v=v, f_sp=f_sp, f_ax=f_ax, f_perp=f_perp, thr_ax=thr_ax, D=D, nz=nz, nmid=nmid)

"""Verifier: burn-time utilization with the AXIAL thrust estimate (f_sp . nose + axial drag), own a_max fit."""
import sys; sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
import numpy as np
from common import *

c = np.load(ROOT / "dev/gnc2_diag/verify_thrust/amax_fit.npy")
def a_max(tau): return 1.0 / (c[0] + c[1] * np.clip(tau, 0, BURN))
def mass(tau): return 1080.0 * (c[0] + c[1] * np.clip(tau, 0, BURN))

def run(kind, name, sims=False):
    d = load(kind, name, sims); x = derive(d, mass)
    am = a_max(x["taum"])
    m = (x["taum"] > 0.2) & (x["taum"] < BURN - 0.05) & np.isfinite(x["thr_ax"])
    U = x["thr_ax"] / am
    Uabs = np.linalg.norm(x["f_sp"], axis=1) / am  # no drag, includes aero normal force
    fz = x["thr_ax"] * x["nmid"][:, 2]
    fh = x["thr_ax"] * np.linalg.norm(x["nmid"][:, :2], axis=1)
    out = dict(n=m.sum(), U=U[m], Uabs=Uabs[m], am=am[m], thr=x["thr_ax"][m], fz=fz[m], fh=fh[m], d=d, x=x, m=m, am_all=am, U_all=U)
    if "A" in d:
        A = d["A"]; g = np.radians(A[:, :2]); cg = np.sqrt(np.clip(1 - np.sin(g[:, 0])**2 - np.sin(g[:, 1])**2, 0, 1))
        out["Umodel"] = (A[:, 3] * cg)[m]; out["thr_cmd"] = A[:, 3][m]
    return out

if __name__ == "__main__":
    lines = []
    for kind in ("their", "ours_strict", "ours_free"):
        R = [run(kind, n) for n in NAMES]
        U = np.concatenate([r["U"] for r in R]); Ua = np.concatenate([r["Uabs"] for r in R])
        am = np.concatenate([r["am"] for r in R]); th = np.concatenate([r["thr"] for r in R])
        fz = np.concatenate([r["fz"] for r in R]); fh = np.concatenate([r["fh"] for r in R])
        s = f"{kind:12s} n={len(U)} U_ax mean {U.mean():.3f} med {np.median(U):.3f} >0.95 {np.mean(U>0.95):.3f} <0.8 {np.mean(U<0.8):.3f} | |f_sp|/amax mean {Ua.mean():.3f}"
        s += f" | per run: avail {am.sum()*DT/11:.1f} used {th.sum()*DT/11:.1f} lost {np.maximum(am-th,0).sum()*DT/11:.1f} m/s | int fz {fz.sum()*DT/11:.1f} int |fh| {fh.sum()*DT/11:.1f} grav {9.81*len(U)*DT/11:.1f}"
        if "Umodel" in R[0]:
            Um = np.concatenate([r["Umodel"] for r in R]); s += f"\n{'':12s} exact model thr*cos(gimbal): mean {Um.mean():.3f} med {np.median(Um):.3f} >0.95 {np.mean(Um>0.95):.3f} <0.8 {np.mean(Um<0.8):.3f}; inferred-exact mean diff {np.mean(U-Um):+.4f}"
        # distinct flights only (drop exact duplicates)
        seen, keep = set(), []
        for r in R:
            key = round(float(np.nansum(r["d"]["R"][:, :3])), 3)
            if key not in seen: seen.add(key); keep.append(r)
        Ud = np.concatenate([r["U"] for r in keep])
        s += f"\n{'':12s} distinct flights {len(keep)}: U_ax mean {Ud.mean():.3f} >0.95 {np.mean(Ud>0.95):.3f}"
        lines.append(s)
    txt = "\n".join(lines); print(txt)
    (ROOT / "dev/gnc2_diag/verify_thrust/util.txt").write_text(txt + "\n")

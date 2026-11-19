"""Launch attitude and quaternion-convention check.
Convention: during the burn the thrust (body z, gimbal <= 15 deg) dominates the specific force, so angle(zb, fsp) must stay
small if R[:,6:10] is scalar-first body->world.  Then inclination/heading of body z at the first live row, the direction of
the displacement over the first 1 s, and the bearing to the first popped balloon."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from vc import KINDS, NAMES, get, rot

def ih(z):
    z = z / np.linalg.norm(z)
    return np.degrees(np.arcsin(z[2])), np.degrees(np.arctan2(z[0], z[1])) % 360

for k in KINDS:
    inc = []
    for n in NAMES:
        d = get(k, n); s = d["s"]; i0 = d["i0"]
        burn = (s > 0.5) & (s < 29.5) & np.isfinite(d["fsp"][:, 0])
        f = d["fsp"][burn]; zb = d["zb"][burn]
        ang = np.degrees(np.arccos(np.clip((f * zb).sum(1) / np.linalg.norm(f, axis=1), -1, 1)))
        # alternative (world->body, i.e. transposed) convention for comparison
        zt = rot(d["R"][burn, 6:10]).transpose(0, 2, 1)[..., :, 2]
        angT = np.degrees(np.arccos(np.clip((f * zt).sum(1) / np.linalg.norm(f, axis=1), -1, 1)))
        I, H = ih(d["zb"][i0])
        j = i0 + 100
        dp = d["R"][j, :3] - d["R"][i0, :3]; Id, Hd = ih(dp)
        fp = d["pops"][0]; ip = int(np.argmin(np.abs(d["t"] - fp[0]))); bb = int(fp[1])
        bpos = d["B"][ip, bb, :3]; brg = np.degrees(np.arctan2(bpos[0] - d["R"][i0, 0], bpos[1] - d["R"][i0, 1])) % 360
        dh = (H - brg + 180) % 360 - 180
        inc.append(I)
        print(f"{k:12s} {n:14s} angle(zb,f) p50 {np.median(ang):5.1f} p95 {np.percentile(ang,95):5.1f} (transposed conv p50 {np.median(angT):5.1f}) | "
              f"launch incl {I:5.1f} head {H:5.1f} | 1 s displacement incl {Id:5.1f} head {Hd:5.1f} | bearing to 1st balloon {brg:5.1f} head-bearing {dh:6.1f}")
    inc = np.array(inc)
    print(f"== {k}: launch inclination median {np.median(inc):.1f} min {inc.min():.1f} max {inc.max():.1f}; tilted (<85) {int((inc < 85).sum())}/11\n")

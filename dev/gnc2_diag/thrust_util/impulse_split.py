"""Where the thrust impulse goes (burn, per run): gravity offset, horizontal, net accel tangential (speed change) vs normal (turning)."""
import sys; sys.path.insert(0, 'dev/gnc2_diag/thrust_util')
import numpy as np
from util_analysis import load, derive, NAMES, DT
for kind in ("their", "ours_strict", "ours_free"):
    acc = {k: 0.0 for k in "avail used fz fh tang tang_pos norm grav_equiv".split()}
    for n in NAMES:
        d = load(kind, n); x = derive(d); m = x["burn"]
        f = x["f_thr"][m]; an = x["f_sp"][m] + np.array([0, 0, -9.81]); v = x["v"][m]
        vh = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-6)
        tang = np.einsum("ij,ij->i", an, vh)
        acc["avail"] += x["am"][m].sum() * DT; acc["used"] += np.linalg.norm(f, axis=1).sum() * DT
        acc["fz"] += f[:, 2].sum() * DT; acc["fh"] += np.linalg.norm(f[:, :2], axis=1).sum() * DT
        acc["tang"] += tang.sum() * DT; acc["tang_pos"] += np.maximum(tang, 0).sum() * DT
        acc["norm"] += np.linalg.norm(an - tang[:, None] * vh, axis=1).sum() * DT
        acc["grav_equiv"] += 9.81 * m.sum() * DT
    print(f"{kind:12s} per run (m/s): avail {acc['avail']/11:.0f} used {acc['used']/11:.0f} | int f_z {acc['fz']/11:.0f} (gravity {acc['grav_equiv']/11:.0f}) "
          f"int |f_h| {acc['fh']/11:.0f} | net accel: tangential sum {acc['tang']/11:+.0f} (speed-up part {acc['tang_pos']/11:.0f}) normal/turning {acc['norm']/11:.0f}")

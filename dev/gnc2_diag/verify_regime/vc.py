"""Independent loader for the verification of the launch_regime finding (does not import launch_regime code)."""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
D = ROOT / "data/gnc2"
NAMES = "10_CoolBa 10_SpaceY 11_Mochi 11_near_miss 12_Apex 13_Apex 13_Yaka 14_SpaceBerry 16_SpaceBerry 16_TeamDJ 18_Taoyuan".split()
KINDS = ("their", "ours_strict", "ours_free")
GV = np.array([0.0, 0.0, -9.81])


def rot(q):
    """Body->world rotation matrix for scalar-first unit quaternion(s) q (..., 4)."""
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], -2)


def get(kind, name):
    z = np.load(D / f"{kind}_{name}.npz")
    d = {k: z[k] for k in z.files}
    R = d["R"].astype(float); t = d["t"].astype(float)
    live = np.isfinite(R[:, 0])
    i0 = int(np.flatnonzero(live)[0])
    d["i0"], d["tL"] = i0, t[i0]
    d["s"] = t - t[i0]  # time since first live row
    v = R[:, 3:6]
    a = np.full_like(v, np.nan)
    a[1:-1] = (v[2:] - v[:-2]) / (t[2:] - t[:-2])[:, None]
    d["acc"] = a
    d["fsp"] = a - GV  # specific force incl. aero (no drag removal)
    d["zb"] = rot(R[:, 6:10])[..., :, 2]
    pops = np.asarray(d["pops"], float).reshape(-1, 2)
    d["pops"] = pops
    d["ps"] = pops[:, 0] - t[i0]
    return d

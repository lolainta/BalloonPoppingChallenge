"""Independent verification helpers (own code, not importing dev/gnc2_diag/attitude)."""
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data/gnc2"
NAMES = "10_CoolBa 10_SpaceY 11_Mochi 11_near_miss 12_Apex 13_Apex 13_Yaka 14_SpaceBerry 16_SpaceBerry 16_TeamDJ 18_Taoyuan".split()
G = np.array([0.0, 0.0, -9.81])
DT = 0.01
BURN = 30.0


def qmul(a, b):
    a0, a1, a2, a3 = np.moveaxis(a, -1, 0); b0, b1, b2, b3 = np.moveaxis(b, -1, 0)
    return np.stack([a0*b0 - a1*b1 - a2*b2 - a3*b3, a0*b1 + a1*b0 + a2*b3 - a3*b2,
                     a0*b2 - a1*b3 + a2*b0 + a3*b1, a0*b3 + a1*b2 - a2*b1 + a3*b0], -1)


def qrot(q, v, conj=False):
    """rotate body vector v by q (scalar-first): q v q*  (conj=True: q* v q)."""
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    qc = q * np.array([1, -1, -1, -1])
    vq = np.concatenate([np.zeros(v.shape[:-1] + (1,)), v], -1)
    vq = np.broadcast_to(vq, q.shape)
    r = qmul(qmul(qc, vq), q) if conj else qmul(qmul(q, vq), qc)
    return r[..., 1:]


def angd(a, b):
    na = np.linalg.norm(a, axis=-1); nb = np.linalg.norm(b, axis=-1)
    c = (a * b).sum(-1) / np.maximum(na * nb, 1e-12)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def get(kind, name):
    d = dict(np.load(DATA / f"{kind}_{name}.npz"))
    t, R = d["t"], d["R"]
    live = ~np.isnan(R[:, 0])
    i0 = int(np.argmax(live)); L = t[i0]
    Rq = np.where(live[:, None], R, 0.0); Rq[~live, 6] = 1.0
    d["live"], d["i0"], d["L"] = live, i0, L
    d["p"], d["v"], d["q"], d["w"] = R[:, :3], R[:, 3:6], Rq[:, 6:10], R[:, 10:13]
    d["zb"] = qrot(d["q"], np.array([0.0, 0.0, 1.0]))
    d["xb"] = qrot(d["q"], np.array([1.0, 0.0, 0.0]))
    d["yb"] = qrot(d["q"], np.array([0.0, 1.0, 0.0]))
    acc = np.full_like(d["v"], np.nan)
    acc[1:-1] = (d["v"][2:] - d["v"][:-2]) / (2 * DT)
    d["acc"] = acc
    d["f"] = acc - G
    d["wxy"] = np.linalg.norm(d["w"][:, :2], axis=1)
    pops = d["pops"]
    d["pops"] = pops[np.argsort(pops[:, 0])] if len(pops) else pops
    d["burnmask"] = live & (t >= L) & (t < L + BURN)
    return d


def ti(t, x):
    return int(np.clip(np.round((x - t[0]) / DT), 0, len(t) - 1))

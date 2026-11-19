"""Verifier: recount pops of the thrust_util sims from their raw S arrays, check base == ours_strict, per-replay deltas,
hop stats, hit relative speed, and duplicate flights."""
import sys; sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
import numpy as np
from common import *
from breakdown import pops_from_S

def stats(kind, sims):
    res = {}
    for n in NAMES:
        d = load(kind, n, sims)
        P = pops_from_S(d)
        R = d["R"]; iL = int(np.flatnonzero(np.isfinite(R[:, 0]))[0]); L = d["t"][iL]
        tp = np.array([p[0] for p in P]) if P else np.zeros(0)
        vrel = [np.linalg.norm(R[i, 3:6] - d["B"][i, b, 3:]) for _, b, i in P]
        res[n] = dict(pops=len(P), burn=int((tp - L <= BURN).sum()), hops=np.diff(np.concatenate([[L], tp])), vrel=vrel,
                      key=round(float(np.nansum(R[:, :3])), 2))
    return res

base = stats("base", True); strict = stats("ours_strict", False)
print("base == ours_strict (pos checksum, pops):", all(base[n]["key"] == strict[n]["key"] and base[n]["pops"] == strict[n]["pops"] for n in NAMES))
lines = []
for v in ["base", "s10", "s08", "fullsat", "fullsat10", "excess2", "noclamp", "lead05", "lead10", "lead15", "lead10f", "freebase", "freelead10"]:
    s = stats(v, True)
    ref = base if not v.startswith("free") else stats("ours_free", False)
    hops = np.concatenate([s[n]["hops"][1:] for n in NAMES])
    vr = np.concatenate([s[n]["vrel"][1:] for n in NAMES if len(s[n]["vrel"]) > 1])
    delta = [s[n]["pops"] - ref[n]["pops"] for n in NAMES]
    keys = {}
    for n in NAMES: keys.setdefault(s[n]["key"], []).append(n)
    dups = [g for g in keys.values() if len(g) > 1]
    lines.append(f"{v:10s} pops {sum(s[n]['pops'] for n in NAMES):3d} inburn {sum(s[n]['burn'] for n in NAMES):3d} hop med {np.median(hops):.2f} mean {hops.mean():.2f} "
                 f"vrel_hit med {np.median(vr):.1f} | delta {delta} better {sum(x>0 for x in delta)} equal {sum(x==0 for x in delta)} worse {sum(x<0 for x in delta)} | identical flights {dups}")
txt = "\n".join(lines); print(txt)
(ROOT / "dev/gnc2_diag/verify_thrust/sims_check.txt").write_text(txt + "\n")

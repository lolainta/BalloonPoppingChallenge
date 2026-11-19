"""Paired decomposition, regression, bend, offset side, lateral accel toward next, TGT handover -- own implementation."""
import numpy as np, pandas as pd
from pathlib import Path
import vhops as V
here = Path(__file__).parent
DUP = {"13_Apex", "16_SpaceBerry"}; FREE = {"10_CoolBa", "10_SpaceY", "11_near_miss", "12_Apex", "14_SpaceBerry"}
th = pd.read_csv(here / "vhops_their.csv"); us = pd.read_csv(here / "vhops_ours_strict.csv"); fr = pd.read_csv(here / "vhops_ours_free.csv")
usd = us[~us.name.isin(DUP)]; frd = fr[fr.name.isin(FREE)]

print("== paired decomposition (dedup strict) ==")
m = th.merge(usd, on=["name", "b0", "b1"], suffixes=("_th", "_us"))
dT = m.T_us - m.T_th
al = (m.Tmin_us - m.Tmin_al_us) - (m.Tmin_th - m.Tmin_al_th)
sp = m.Tmin_al_us - m.Tmin_al_th
ex = (m.T_us - m.Tmin_us) - (m.T_th - m.Tmin_th)
se = lambda x: x.std() / np.sqrt(len(x))
print(f"n={len(m)} dT mean {dT.mean():+.3f} med {dT.median():+.3f} se {se(dT):.3f} | align {al.mean():+.3f} (se {se(al):.3f}) | speed {sp.mean():+.3f} (se {se(sp):.3f}) | exec {ex.mean():+.3f} (se {se(ex):.3f})")
# which hops pair? are the unpaired ones different?
print("paired their T median", m.T_th.median(), "all their", th["T"].median(), "| paired ours", m.T_us.median(), "all ours", usd["T"].median())
# how many of the ours_strict hops are consecutive pops of their order (strict => every hop should be (b_k,b_k+1) of their list unless skipped)
print("ours_strict hops that pair:", len(m), "of", len(usd))

print("\n== regression T ~ d + vclose + vperp + ours ==")
def ols(y, X, names):
    X1 = np.column_stack([np.ones(len(y)), X]); b, *_ = np.linalg.lstsq(X1, y, rcond=None)
    res = y - X1 @ b; s2 = res @ res / (len(y) - X1.shape[1]); s = np.sqrt(np.diag(s2 * np.linalg.inv(X1.T @ X1)))
    r2 = 1 - res @ res / ((y - y.mean()) @ (y - y.mean()))
    print("  n %d R2 %.2f " % (len(y), r2) + " ".join(f"{n} {bi:+.3f}±{si:.3f}" for n, bi, si in zip(["c"] + names, b, s)))
    return b
for lab, o in [("strict dedup", usd), ("strict all(dups)", us), ("free dedup", frd), ("free all(dups)", fr)]:
    P = pd.concat([th.assign(o=0), o.assign(o=1)]); P = P[np.isfinite(P.Tmin)]
    b = ols(P["T"].values, P[["d", "vclose", "vperp", "o"]].values, ["d", "vc", "vp", "ours"])
    cf = b[2] * (th.vclose.median() - o.vclose) + b[3] * (th.vperp.median() - o.vperp)
    print(f"  {lab}: counterfactual mean {cf.mean():+.2f} s; raw mean gap {o['T'].mean() - th['T'].mean():+.2f}")
    # alternative: exclude very short hops (<0.5 s, double pops)
    P2 = P[P["T"] > 0.5]
    ols(P2["T"].values, P2[["d", "vclose", "vperp", "o"]].values, ["d", "vc", "vp", "ours"])

print("\n== bend (arriving chord vs LOS to next) and rotation ==")
def bend(h):
    out = []
    for n, g in h.groupby("name"):
        g = g.sort_values("k"); P = g[["px", "py", "pz"]].values; Rl = g[["rx", "ry", "rz"]].values; Vv = g[["vx", "vy", "vz"]].values
        for i in range(1, len(g)):
            ch = P[i] - P[i - 1]
            out.append((V.angd(ch, Rl[i]), g.th.values[i]))
    a = np.array(out); return np.nanmedian(a[:, 0]), np.nanmedian(a[:, 1] - a[:, 0]), len(a)
for lab, h in [("their", th), ("strict", usd), ("free", frd)]:
    print(lab, "bend %.1f  th-bend %.1f  n %d" % bend(h))

"""How much hop time does the arrival geometry explain? OLS of hop time on geometry at the pop, pooled their + ours;
then counterfactual: our hops with their median alignment."""
import numpy as np, pandas as pd
from pathlib import Path
here = Path(__file__).parent; D = here.parents[2] / "data/gnc2"
H = {}
for kind in ("their", "ours_strict", "ours_free"):
    h = pd.read_csv(here / f"hops_{kind}.csv"); h = h[(h.k > 0) & np.isfinite(h.Tmin)].copy(); h["kind"] = kind
    if kind != "their":
        st = []
        for _, r in h.iterrows():
            z = np.load(D / f"{kind}_{r['name']}.npz"); s = (z["t"] >= r.t0) & (z["t"] < r.t0 + r["T"]); st.append(np.median(z["STRETCH"][s]))
        h["stretch"] = st
    else:
        h["stretch"] = np.nan
    H[kind] = h


def ols(y, X, names):
    X1 = np.column_stack([np.ones(len(y)), X]); b, *_ = np.linalg.lstsq(X1, y, rcond=None)
    res = y - X1 @ b; s2 = res @ res / (len(y) - X1.shape[1]); se = np.sqrt(np.diag(s2 * np.linalg.inv(X1.T @ X1)))
    r2 = 1 - res @ res / ((y - y.mean()) @ (y - y.mean()))
    print("   R2 %.2f  " % r2 + "  ".join(f"{n} {bi:+.3f}±{si:.3f}" for n, bi, si in zip(["c"] + names, b, se)))
    return b


for kind in ("their", "ours_strict", "ours_free"):
    h = H[kind]
    print(kind, "n", len(h))
    # T ~ d, (1 - cos th_int) * vrel, vclose   (geometry only)
    print("  T ~ d + vclose + vperp")
    ols(h["T"].to_numpy(), h[["d", "vclose", "vperp"]].to_numpy(), ["d", "vclose", "vperp"])
    print("  T/Tmin ~ nose_err + th_int")
    ols((h["T"] / h.Tmin).to_numpy(), h[["nose_err", "th_int"]].to_numpy(), ["nose_err", "th_int"])
    if kind != "their":
        hh = h[h.stretch == 1.0]
        print("  stretch==1.0 only: T/Tmin ~ nose_err + th_int  (n=%d)" % len(hh))
        ols((hh["T"] / hh.Tmin).to_numpy(), hh[["nose_err", "th_int"]].to_numpy(), ["nose_err", "th_int"])
# pooled their + ours_strict: T ~ d + vclose + vperp + ours indicator
P = pd.concat([H["their"], H["ours_strict"]])
print("pooled their+ours_strict: T ~ d + vclose + vperp + is_ours")
b = ols(P["T"].to_numpy(), np.column_stack([P[["d", "vclose", "vperp"]].to_numpy(), (P.kind != "their").to_numpy(float)]), ["d", "vclose", "vperp", "ours"])
us, th = H["ours_strict"], H["their"]
cf = (b[2] * (th.vclose.median() - us.vclose) + b[3] * (th.vperp.median() - us.vperp))
print("  counterfactual: our hops with their median closing / perpendicular speed at the pop -> T change mean %.2f s" % cf.mean())
P = pd.concat([H["their"], H["ours_free"]])
print("pooled their+ours_free: T ~ d + vclose + vperp + is_ours")
b = ols(P["T"].to_numpy(), np.column_stack([P[["d", "vclose", "vperp"]].to_numpy(), (P.kind != "their").to_numpy(float)]), ["d", "vclose", "vperp", "ours"])

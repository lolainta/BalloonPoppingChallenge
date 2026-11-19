"""Summarize sims/<variant>_<name>.npz: pops, hop times, utilization, per-hop bounds (same definitions as util_analysis)."""
import sys, glob; sys.path.insert(0, 'dev/gnc2_diag/thrust_util')
import numpy as np
from util_analysis import derive, hop_stats, NAMES, DT
from pathlib import Path
D = Path('dev/gnc2_diag/thrust_util/sims')
variants = sys.argv[1:] or sorted({p.name.split('_')[0] for p in D.glob('*_*.npz')})
base_pops = {}
print(f"{'variant':10s} {'runs':>4s} {'pops':>5s} {'burnpops':>8s} {'hop_med':>7s} {'hop_mean':>8s} {'first_med':>9s} {'U_mean':>6s} {'thr_mean':>8s} "
      f"{'T_lb':>5s} {'magL':>5s} {'dirL':>5s} {'vrel':>5s} {'dPops_vs_base':>13s}")
for v in variants:
    files = [D / f"{v}_{n}.npz" for n in NAMES if (D / f"{v}_{n}.npz").exists()]
    if not files:
        continue
    pops = burnp = 0; U = []; thr = []; hops = []; first = []; per = {}
    for f in files:
        d = {k: val for k, val in np.load(f).items()}; x = derive(d); m = x["burn"]
        pt = d["pops"][:, 0] if len(d["pops"]) else np.array([])
        pops += len(pt); burnp += int((pt <= x["L"] + 30).sum()); per[f.stem.split('_', 1)[1]] = len(pt)
        U.append(x["U"][m]); thr.append(d["A"][m, 3])
        hs = hop_stats(v, f.stem, d, x)
        hops += [h for h in hs if not h["first"] and h["burn_frac"] > 0.5 and np.isfinite(h["T_mag"])]
        first += [h["T_act"] for h in hs if h["first"]]
    a = lambda k: np.array([h[k] for h in hops])  # noqa: E731
    if v == "base":
        base_pops = per
    dp = " ".join(f"{per[n] - base_pops.get(n, per[n]):+d}" for n in NAMES if n in per) if base_pops else ""
    print(f"{v:10s} {len(files):4d} {pops:5d} {burnp:8d} {np.median(a('T_act')):7.2f} {a('T_act').mean():8.2f} {np.median(first):9.2f} "
          f"{np.concatenate(U).mean():6.3f} {np.concatenate(thr).mean():8.3f} {a('T_lb').mean():5.2f} {(a('T_mag') - a('T_lb')).mean():5.2f} "
          f"{(a('T_act') - a('T_mag')).mean():5.2f} {np.median(a('rel_speed_hit')):5.1f}  {dp}")

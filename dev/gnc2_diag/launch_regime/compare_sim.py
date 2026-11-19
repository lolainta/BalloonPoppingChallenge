"""Summarise sim/<variant>_<mode>_<name>.npz: totals, first pop, cumulative pops by t+10 s (launch phase), by burnout, after.
Per-run deltas vs base; noise floor from the j* jitter variants (89.5 deg launch, 4 headings).
.venv/bin/python dev/gnc2_diag/launch_regime/compare_sim.py strict base,att,low68,nodown,know,j0,j90,j180,j270
"""
import sys
from pathlib import Path
import numpy as np

D = Path(__file__).resolve().parent / "sim"
NAMES = "10_CoolBa 10_SpaceY 11_Mochi 11_near_miss 12_Apex 13_Apex 13_Yaka 14_SpaceBerry 16_SpaceBerry 16_TeamDJ 18_Taoyuan".split()
mode, variants = sys.argv[1], sys.argv[2].split(",")
tab = {}
for v in variants:
    rows = []
    for n in NAMES:
        f = D / f"{v}_{mode}_{n}.npz"
        if not f.exists():
            rows.append([np.nan] * 5); continue
        d = np.load(f); p = d["pops"][:, 0] - float(d["launch"]) if len(d["pops"]) else np.array([])
        rows.append([len(p), p[0] if len(p) else np.nan, (p <= 10).sum(), (p <= 30).sum(), (p > 30).sum()])
    tab[v] = np.array(rows, float)
print(f"{'variant':8s} {'total':>6s} {'first mean':>10s} {'first med':>9s} {'<=10s':>6s} {'<=30s':>6s} {'>30s':>5s}   per-run pops")
for v, A in tab.items():
    print(f"{v:8s} {np.nansum(A[:,0]):6.0f} {np.nanmean(A[:,1]):10.2f} {np.nanmedian(A[:,1]):9.2f} {np.nansum(A[:,2]):6.0f} {np.nansum(A[:,3]):6.0f} {np.nansum(A[:,4]):5.0f}   "
          + " ".join(f"{x:2.0f}" for x in A[:, 0]))
J = [v for v in tab if v.startswith("j")]
if J and "base" in tab:
    tots = np.array([np.nansum(tab[v][:, 0]) for v in J] + [np.nansum(tab["base"][:, 0])])
    print(f"\nnoise floor (base + {len(J)} jitters): totals {tots.astype(int).tolist()}  mean {tots.mean():.1f}  SD {tots.std(ddof=1):.1f}")
    per = np.stack([tab[v][:, 0] for v in J] + [tab["base"][:, 0]])
    print(f"per-run SD mean {np.nanmean(per.std(0, ddof=1)):.2f} pops; first-pop SD mean {np.nanmean(np.stack([tab[v][:,1] for v in J]+[tab['base'][:,1]]).std(0, ddof=1)):.2f} s")

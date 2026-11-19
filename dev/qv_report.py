"""Gates for the critic re-ranking probe (agents/rl3_q.py) from a shadow run's qv_log CSVs (+ the sweep CSVs of that run).
   python dev/qv_report.py --log runs/qv/shadow_X [--sweep runs/.cmp_parts/probe_X/hour*.csv] [--gamma 0.99] [--dt 0.5]
(a) corr(V1, realized return after the decision), the return built as the critic's training target: reward = pops per row,
    discount GAMMA ** (dt_row / 0.5 s) (RL3_FAST), coast pops after burnout credited at burnout (dev/rl2_env.py), no time cost
    (RL3_TCOST off on s40k-n), no scaling: R(t) = sum over pops at t_p > t of gamma ** ((min(t_p, t_bo) - t) / dt)
(b) confident decisions (gap > 3): fraction with V2 - V1 > delta (false alarms)
(c) eligible decisions and would-be overrides per episode over qv_gap x qv_delta (capped: the first per episode, qv_max 1)
"""
import argparse, csv, glob
from collections import defaultdict

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--log", nargs="+", required=True, help="qv_log prefixes (or CSV files / globs)")
ap.add_argument("--sweep", nargs="*", default=[], help="sweep_gnc --csv files of the same run (episode counts, pop check)")
ap.add_argument("--gamma", type=float, default=0.99); ap.add_argument("--dt", type=float, default=0.5)
ap.add_argument("--conf", type=float, default=3.0)
a = ap.parse_args()
files = sorted({f for p in a.log for f in (glob.glob(p) if p.endswith(".csv") else glob.glob(f"{p}.*.csv"))})
assert files, f"no qv_log CSVs for {a.log}"
dec, pops, t_bo, key, walled = defaultdict(list), defaultdict(list), {}, {}, set()
for f in files:
    for r in csv.DictReader(open(f)):
        run = r["run"]; key[run] = (int(r["seed"]), int(r["cfg"]))
        if r["kind"] == "pop":
            pops[run].append(float(r["t"]))
        elif r["kind"] == "wall":
            walled.add(run)
        else:
            dec[run].append(r); t_bo[run] = float(r["t_bo"])
runs = sorted(key)
dup = len(runs) - len(set(key.values()))
cfgs = sorted({key[r][1] for r in runs})
print(f"{len(files)} log files, {len(runs)} episodes (cfg {cfgs}), {sum(map(len, dec.values()))} decisions, "
      f"{sum(map(len, pops.values()))} pops, {len(walled)} hit qv_wall" + (f", {dup} duplicate (seed, cfg) episodes" if dup else ""))
n_ep = len(runs)
if a.sweep:
    sw = {}
    for f in a.sweep:
        for r in csv.DictReader(open(f)):
            sw[int(r["seed"]), int(r["cfg"])] = int(float(r["pops"]))
    mine = {k: v for k, v in sw.items() if k[1] in cfgs}
    n_ep = len(mine)
    got = defaultdict(int)
    for r in runs:
        got[key[r]] = len(pops[r])
    bad = [k for k, v in mine.items() if got.get(k) != v]
    print(f"sweep: {n_ep} episodes of cfg {cfgs}, mean pops {np.mean(list(mine.values())):.3f}; pop rows != sweep pops: {len(bad)}"
          + (f" e.g. {bad[:5]}" if bad else ""))
    for c in sorted({k[1] for k in sw}):
        print(f"  cfg {c}: mean pops {np.mean([v for k, v in sw.items() if k[1] == c]):.3f} ({sum(k[1] == c for k in sw)} ep)")

rows = []  # (run, t, gap, V1, V2, R)
for r in runs:
    tp = np.array(pops[r])
    for d in dec[r]:
        t = float(d["t"])
        f = tp[tp > t + 1e-9]
        R = float((a.gamma ** ((np.minimum(f, t_bo[r]) - t) / a.dt)).sum())
        rows.append((r, t, float(d["gap"]), float(d["V1"]), float(d["V2"]), R, int(d["eligible"])))
assert rows, "no decision rows"
run_ = np.array([x[0] for x in rows]); T, G, V1, V2, R, EL = (np.array([x[i] for x in rows], float) for i in range(1, 7))
dV = V2 - V1


def rank(x):
    o = np.argsort(x, kind="stable"); r = np.empty(len(x)); r[o] = np.arange(len(x)); return r


print(f"\nV scale: V1 mean {V1.mean():.3f} std {V1.std():.3f} | realized R mean {R.mean():.3f} std {R.std():.3f}")
for g in (0.5, 1.0, 2.0):
    m = G < g
    if m.any():
        print(f"  V2 - V1 on gap < {g}: n {m.sum()}  mean {dV[m].mean():+.3f}  std {dV[m].std():.3f}  "
              f"p90 {np.quantile(dV[m], 0.9):+.3f}")
if EL.any():
    print(f"  V2 - V1 on the logged eligible flag: n {int(EL.sum())}  mean {dV[EL > 0].mean():+.3f}  std {dV[EL > 0].std():.3f}")

print(f"\n(a) critic vs realized return (gamma {a.gamma} per {a.dt} s, coast pops at burnout), n {len(R)}")
for name, m in (("all", np.ones(len(R), bool)), ("gap < 1", G < 1.0), ("gap < 2", G < 2.0), (f"gap > {a.conf:g}", G > a.conf)):
    if m.sum() > 2:
        print(f"  {name:>8}: n {m.sum():5d}  pearson {np.corrcoef(V1[m], R[m])[0, 1]:.3f}  spearman "
              f"{np.corrcoef(rank(V1[m]), rank(R[m]))[0, 1]:.3f}  bias V1-R {np.mean(V1[m] - R[m]):+.3f}  rmse {np.sqrt(np.mean((V1[m] - R[m]) ** 2)):.3f}")
for lo, hi in ((0, 20), (20, 40), (40, 60), (60, 1e9)):  # (the late decisions are where the returns are small)
    m = (T >= lo) & (T < hi)
    if m.sum() > 2:
        print(f"  t {lo:>3}-{hi if hi < 1e9 else 'end':>3} s: n {m.sum():5d}  pearson {np.corrcoef(V1[m], R[m])[0, 1]:.3f}  "
              f"mean V1 {V1[m].mean():.3f} R {R[m].mean():.3f}")

m = G > a.conf
print(f"\n(b) confident decisions (gap > {a.conf:g}): n {m.sum()} ({m.mean():.1%} of decisions)")
for dl in (0.1, 0.2, 0.3):
    print(f"  delta {dl}: false alarms V2 - V1 > delta {np.mean(dV[m] > dl) if m.any() else float('nan'):.1%}")

print(f"\n(c) per episode ({n_ep} episodes; decision rows = j1 != held target {len(R) / n_ep:.2f}/ep)")
print("  qv_gap  eligible/ep | qv_delta: would-override/ep (uncapped) / episodes with >= 1 (= capped, qv_max 1)")
for g in (0.5, 1.0, 2.0):
    e = G < g
    parts = []
    for dl in (0.1, 0.2, 0.3):
        o = e & (dV > dl)
        parts.append(f"{dl}: {o.sum() / n_ep:.3f} / {len(set(run_[o])) / n_ep:.3f}")
    print(f"  {g:>5}  {e.sum() / n_ep:9.3f}   | " + "  ".join(parts))

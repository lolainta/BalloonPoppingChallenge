"""Report for approach_fork.py. .venv/bin/python dev/gnc2_seq/approach/report.py [APPRTAG [CHAOSTAG]]
policy (free) vs best-found (paired), chaos-control gain, net upper bound (best_appr - best_chaos per seed), gain
distribution, options used by the best routes, velocity angle to the next popped balloon at each pop."""
import json, sys
from collections import Counter
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parent / "out"
OPTS = ["none", "h+3", "h-3", "h+6", "h-6", "v+4", "v-4", "to2"]
at, ct = (sys.argv[1:] + ["appr08", "chaos"])[:2] if len(sys.argv) > 1 else ("appr08", "chaos")
if len(sys.argv) == 2:
    ct = "chaos"
load = lambda t: {r["seed"]: r for r in map(json.loads, open(OUT / f"{t}.jsonl")) if "error" not in r} if (OUT / f"{t}.jsonl").exists() else {}
free = {r["seed"]: r for r in map(json.loads, open(OUT / "free.jsonl")) if r.get("mode") == "free" and "error" not in r}
A, C = load(at), load(ct)
ms = lambda x: f"{np.mean(x):+.3f} +- {np.std(x, ddof=1) / np.sqrt(len(x)):.3f} (n={len(x)})" if len(x) > 1 else str(x)
fp = lambda s: len(free[s]["pops"])


def block(name, R):
    ss = sorted(s for s in R if s in free)
    exact = sum(R[s]["greedy"] is not None and R[s]["greedy"]["route"]["pops"] == free[s]["pops"] for s in ss)
    g = [R[s]["best"]["pops"] - fp(s) for s in ss]
    print(f"== {name}: n={len(ss)}  greedy==free {exact}/{len(ss)}  trunc {sum(R[s]['trunc'] for s in ss)}  "
          f"legs/seed {np.mean([R[s]['legs'] for s in ss]):.0f}")
    print(f"   policy {np.mean([fp(s) for s in ss]):.3f}  best {np.mean([R[s]['best']['pops'] for s in ss]):.3f}  "
          f"gain {ms(g)}  dist {dict(sorted(Counter(g).items()))}")
    return ss


sa, sc = block(f"approach search [{at}]", A), block(f"chaos control [{ct}]", C)
both = sorted(set(sa) & set(sc))
if len(both) > 1:
    print(f"== on the {len(both)} seeds with both: search gain {ms([A[s]['best']['pops'] - fp(s) for s in both])}  "
          f"chaos gain {ms([C[s]['best']['pops'] - fp(s) for s in both])}")
    net = [A[s]["best"]["pops"] - C[s]["best"]["pops"] for s in both]
    print(f"   NET upper bound (best_appr - best_chaos, paired) {ms(net)}  dist {dict(sorted(Counter(net).items()))}")
print("   per seed (seed, policy, appr best, chaos best):", [(s, fp(s), A[s]["best"]["pops"] if s in A else None,
      C[s]["best"]["pops"] if s in C else None) for s in sorted(set(sa) | set(sc))])

# options used by the best approach routes (per branch point; option 0 = policy's normal flight)
nz = [sum(j != 0 for j in A[s]["best"]["ranks"]) for s in sa]
cnt = Counter(OPTS[j] for s in sa for j in A[s]["best"]["ranks"])
nbp = sum(len(A[s]["best"]["ranks"]) for s in sa)
print(f"== best approach routes: {nbp} branch points, non-default options per seed mean {np.mean(nz):.2f} "
      f"(dist {dict(sorted(Counter(nz).items()))}); share by option: " +
      ", ".join(f"{k} {v / nbp:.3f}" for k, v in sorted(cnt.items(), key=lambda kv: -kv[1])))
gain_s = [s for s in sa if A[s]["best"]["pops"] > fp(s)]
if gain_s:
    cg = Counter(OPTS[j] for s in gain_s for j in A[s]["best"]["ranks"] if j)
    print(f"   on the {len(gain_s)} seeds with a gain: non-default options per seed {np.mean([sum(j != 0 for j in A[s]['best']['ranks']) for s in gain_s]):.2f}; {dict(cg)}")


def angles(route, inburn=True):
    be = route["burn_end"] or np.inf; P = route["pops"]
    return [a for i, a in enumerate(route["ang"]) if a is not None and (not inburn or P[i + 1][0] < be)]


def angs_by_opt(route):
    """angle at pop i paired with the option flown on the leg that ARRIVED at pop i."""
    be = route["burn_end"] or np.inf; P = route["pops"]
    return [(route["opts"][i], a) for i, a in enumerate(route["ang"]) if a is not None and P[i + 1][0] < be]


print("== velocity angle at a pop to the NEXT popped balloon (both pops in the burn), deg: median / mean")
for name, get, ss in (("policy (free)", lambda s: free[s], sa), ("best approach", lambda s: A[s]["best"]["route"], sa),
                      ("best chaos", lambda s: C[s]["best"]["route"], sc)):
    pool = np.concatenate([angles(get(s)) for s in ss if angles(get(s))]) if ss else np.array([np.nan])
    print(f"   {name:14s}: {np.median(pool):5.1f} / {np.mean(pool):5.1f}  hops {len(pool)}  share <30 deg {np.mean(pool < 30):.2f}")
pd = [np.mean(angles(A[s]["best"]["route"])) - np.mean(angles(free[s])) for s in sa if angles(A[s]["best"]["route"]) and angles(free[s])]
if len(pd) > 1:
    print(f"   paired per-seed mean angle, best approach - policy: {ms(pd)}")
bo = [x for s in sa for x in angs_by_opt(A[s]["best"]["route"])]
if bo:
    print("   best approach routes, angle by the option flown into that pop: " + ", ".join(
        f"{OPTS[o]} {np.median([a for oo, a in bo if oo == o]):.0f} (n={sum(oo == o for oo, _ in bo)})" for o in sorted({o for o, _ in bo})))

"""Paired per-replay comparison of strict variants vs base: launch->first pop, time of the n-th pop (n = common count),
pops in the burn. 12_Apex / 13_Apex are the same flight (same seed and order) -> counted once."""
import sys
import numpy as np
from simsum import stats, NAMES

U = [n for n in NAMES if n != "13_Apex"]
base = {n: stats("base", n) for n in U}
print(f"{'variant':<15}{'d first (s)':>22}{'d t_n / n (s/hop)':>22}{'d pops burn':>16}   first per replay (base -> variant)")
for v in sys.argv[1:]:
    S = {n: stats(v, n) for n in U}
    if any(s is None for s in S.values()):
        continue
    df = np.array([S[n]["first"] - base[n]["first"] for n in U])
    dh = []
    for n in U:
        k = min(len(base[n]["pt"]), len(S[n]["pt"]))
        if k >= 2:  # per-hop shift of the n-th pop excluding the first hop
            dh.append(((S[n]["pt"][k - 1] - S[n]["pt"][0]) - (base[n]["pt"][k - 1] - base[n]["pt"][0])) / (k - 1))
    dh = np.array(dh)
    dp = np.array([S[n]["pb"] - base[n]["pb"] for n in U])
    se = lambda x: x.std(ddof=1) / np.sqrt(len(x))
    print(f"{v:<15}{df.mean():+10.2f} +- {se(df):4.2f} (med {np.median(df):+5.2f}){dh.mean():+10.3f} +- {se(dh):5.3f}{dp.mean():+8.2f} +- {se(dp):4.2f}   "
          + " ".join(f"{base[n]['first']:.1f}>{S[n]['first']:.1f}" for n in U))

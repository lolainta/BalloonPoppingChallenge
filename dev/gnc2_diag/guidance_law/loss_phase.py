"""Per-hop phase summary of loss_profile.pkl (dedup): loss accumulated by 0.5 s / 1.0 s into the hop and by 0.5 s before
the pop, final excess, pointing errors, time to align the actual specific force within 10 deg of the ideal direction."""
import pickle
import numpy as np, pandas as pd
from pathlib import Path

pd.set_option("display.width", 250)
p = pickle.load(open(Path(__file__).with_name("loss_profile.pkl"), "rb"))
p = p[~p.dup]


def phase(g):
    g = g.sort_values("tau"); hop = g.hop.iloc[0]
    Lt = lambda tau: np.interp(tau, g.tau, g.loss)  # noqa: E731
    return pd.Series(dict(hop=hop, T0=g.T0.iloc[0], L05=Lt(0.5), L10=Lt(1.0), Lrem05=Lt(max(hop - 0.5, 0)), Lend=hop - g.T0.iloc[0],
                          ang0=g.ang_body.iloc[0], angf0=g.ang_force.iloc[0], ang_mid=g[(g.tau > 0.5) & (g.rem > 0.5)].ang_force.median(),
                          rate_max=g.rate.max(), t_align=(g.tau[g.ang_force < 10].min() if (g.ang_force < 10).any() else np.nan)))


res = p.groupby(["kind", "name", "k"]).apply(phase).reset_index()
res = res[np.isfinite(res.T0)]
pickle.dump(res, open(Path(__file__).with_name("loss_phase.pkl"), "wb"))
cols = ["hop", "T0", "L05", "L10", "Lrem05", "Lend", "ang0", "angf0", "ang_mid", "rate_max", "t_align"]
print("MEAN (all dedup hops)"); print(res.groupby("kind")[cols].mean().round(2).to_string())
print("MEDIAN"); print(res.groupby("kind")[cols].median().round(2).to_string())
short = res[res.hop < 8]
print("MEAN (hops < 8 s: without the two long re-pass hops)"); print(short.groupby("kind")[cols].mean().round(2).to_string())
print("n", res.groupby("kind").size().to_dict())

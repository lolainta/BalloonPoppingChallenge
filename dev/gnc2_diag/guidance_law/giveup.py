"""Terminal give-up artefact: our GNC's reach time asks for an exact centre hit (tol 0). In the last few tenths of a
second an off-centre pass (still inside the 1.5 m radius) becomes 'unreachable' -> TGO jumps to a turn-around solution
(seconds) and the guidance swings the thrust toward it just before the hit. Count, per pop: TGO jump (> +0.5 s) of the
target within the last 0.6 s, the commanded-force swing (angle between F before the jump and F at the pop), the body
rate at the pop, and the start-of-hop loss L05 of the following hop (loss_phase.pkl) with vs without a jump."""
import pickle
import numpy as np, pandas as pd
from chases import D, NAMES
from hops import SEED

ph = pickle.load(open("loss_phase.pkl", "rb"))
rows, seen = [], set()
for n in NAMES:
    for kind in ("ours_strict", "ours_free"):
        f = np.load(D / f"{kind}_{n}.npz"); t, TGO, TGT, F, R, L = f["t"], f["TGO"], f["TGT"], f["F"], f["R"], float(f["launch"])
        pops = f["pops"]
        for k, (tp, b) in enumerate(pops):
            b = int(b); key = (kind, SEED[n], b, round(float(tp), 2))
            if key in seen or tp > L + 29.5:
                continue
            seen.add(key)
            i1 = int(np.searchsorted(t, tp - 1e-6))
            w = np.arange(max(i1 - 60, 1), i1)
            w = w[TGT[w] == b]
            if len(w) < 5:
                continue
            jumps = w[1:][np.diff(TGO[w]) > 0.5]
            jump = len(jumps) > 0
            swing = np.nan
            if jump:
                j = jumps[0]
                fa, fb = F[j - 1], F[i1 - 1]
                swing = np.degrees(np.arccos(np.clip(fa @ fb / (np.linalg.norm(fa) * np.linalg.norm(fb)), -1, 1)))
            nxt = ph[(ph.kind == kind) & (ph.name == n) & (ph.k == k + 1)]
            rows.append(dict(kind=kind, name=n, k=k, jump=jump, t_before=(tp - t[jumps[0]]) if jump else np.nan, swing=swing,
                             rate=np.linalg.norm(R[i1, 10:13]), L05_next=nxt.L05.iloc[0] if len(nxt) else np.nan,
                             L10_next=nxt.L10.iloc[0] if len(nxt) else np.nan, ang0_next=nxt.ang0.iloc[0] if len(nxt) else np.nan))
r = pd.DataFrame(rows)
pd.set_option("display.width", 200)
for kind in ("ours_strict", "ours_free"):
    x = r[r.kind == kind]
    print(f"{kind}: pops {len(x)}, with terminal TGO jump {int(x.jump.sum())} ({100 * x.jump.mean():.0f}%), jump {x.t_before.median():.2f} s before the pop (median), "
          f"commanded-force swing median {x.swing.median():.0f} deg")
    print(x.groupby("jump").agg(n=("rate", "size"), rate_at_pop=("rate", "mean"), L05_next=("L05_next", "mean"), L10_next=("L10_next", "mean"),
                                ang0_next=("ang0_next", "median")).round(2).to_string())
pickle.dump(r, open("giveup.pkl", "wb"))

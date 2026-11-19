"""Free mode: can GNC predict the policy's next target before the pop? At the state 0.5 s before pop k, rank the alive
balloons (excluding the current one) by min reach time from the predicted arrival (state at the pop), compare with the
target the policy actually picks right after the pop (TGT)."""
import numpy as np
import hops as H
FREE = ["10_CoolBa", "10_SpaceY", "11_near_miss", "12_Apex", "14_SpaceBerry"]
top1 = top3 = n = 0
for nm in FREE:
    z = np.load(H.D / f"ours_free_{nm}.npz"); t, R, B, S, TGT = z["t"], z["R"], z["B"], z["S"], z["TGT"]
    launch = float(z["launch"])
    P = [(float(tp), int(b)) for tp, b in z["pops"] if tp <= launch + 29]
    for tp, b in P:
        i = int(np.argmin(np.abs(t - tp)))
        after = TGT[i:i + 60]; after = after[(after >= 0) & (after != b)]
        if not len(after):
            continue
        nxt = int(after[0])
        p, v = R[i - 1, :3], R[i - 1, 3:6]  # arrival state (what a predictor would aim for)
        alive = np.flatnonzero((S[i - 1] == 1)); alive = alive[alive != b]
        tm = np.array([H.tmin(p, v, B[i - 1, a, :3].astype(float), B[i - 1, a, 3:].astype(float), 13.0, 0.0)[0] for a in alive])
        order = alive[np.argsort(tm)]
        n += 1; top1 += order[0] == nxt; top3 += nxt in order[:3]
print(f"free-mode pops {n}: policy's next target = min-reach-time balloon from the arrival state: top1 {top1/n:.2f}, top3 {top3/n:.2f}")

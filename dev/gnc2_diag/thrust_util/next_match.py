"""How often is the min-reach-time 'predicted next target' (used by freelead10) the balloon our policy actually chases next?
Evaluated 0.5 s before each pop in the logged free runs (data/gnc2/ours_free_*), drift-free reach time at the logged a_thrust."""
import sys; sys.path.insert(0, '.')
import numpy as np
from BalloonPoppingGymEnv.agents.rl2_agent import reach_time
NAMES = "10_CoolBa 10_SpaceY 11_Mochi 11_near_miss 12_Apex 13_Apex 13_Yaka 14_SpaceBerry 16_SpaceBerry 16_TeamDJ 18_Taoyuan".split()
grid = np.arange(0.1, 15.0, 0.1)
hit = tot = 0
for kind in ("ours_free", "ours_strict"):
    hit = tot = 0
    for n in NAMES:
        d = np.load(f"data/gnc2/{kind}_{n}.npz"); t, R, B, S, TGT = d["t"], d["R"], d["B"], d["S"], d["TGT"]
        pops = d["pops"]
        for (tp, b), (tq, b2) in zip(pops[:-1], pops[1:]):
            i = int(np.searchsorted(t, tp)) - 50
            if i < 0 or TGT[i] != int(b):
                continue
            j = int(np.searchsorted(t, tp)) + 20  # the target actually chased right after the pop
            nxt = TGT[min(j, len(t) - 1)]
            if nxt < 0:
                continue
            tgo = tp - t[i]
            p, v = R[i, :3], R[i, 3:6]
            alive = np.flatnonzero((S[i] == 1) & (np.arange(S.shape[1]) != int(b)))
            p_arr = B[i, int(b), :3] + B[i, int(b), 3:] * tgo
            t2 = reach_time(p_arr, v, B[i, alive, :3] + B[i, alive, 3:] * tgo, B[i, alive, 3:], float(d["a_thrust"][i]), grid)
            hit += int(alive[int(np.argmin(t2))] == nxt); tot += 1
    print(f"{kind}: predicted next == actually chased next in {hit}/{tot} = {hit / max(tot, 1):.2f}")

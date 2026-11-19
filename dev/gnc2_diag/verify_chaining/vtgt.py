"""Does our free policy switch away from a balloon before it pops? TGT at the pop step and the step before; time of the
last step with TGT == popped balloon; switches away from a live target and whether it popped later."""
import numpy as np
from pathlib import Path
import vhops as V
DUP = {"13_Apex", "16_SpaceBerry"}; FREE = {"10_CoolBa", "10_SpaceY", "11_near_miss", "12_Apex", "14_SpaceBerry"}
for kind, keep in [("ours_free", FREE), ("ours_strict", set(V.NAMES) - DUP)]:
    n_pop = n_away = 0; lead = []; sw_live = sw_pop2 = 0; far = []; n_no_tgt = 0; n_never = 0
    for n in sorted(keep):
        z = np.load(V.D / f"{kind}_{n}.npz"); t, S, TGT, R, B = z["t"], z["S"], z["TGT"], z["R"], z["B"]
        pops = sorted((int(np.flatnonzero(S[:, b] == 2)[0]), b) for b in range(100) if (S[:, b] == 2).any())
        for i, b in pops:
            n_pop += 1
            if TGT[i - 1] != b:
                n_away += 1
                w = np.flatnonzero(TGT[:i] == b)
                if len(w): lead.append(t[i] - t[w[-1]])
                else: n_never += 1
        # switches away from a live target
        for s in range(1, len(t)):
            a0 = TGT[s - 1]
            if a0 >= 0 and TGT[s] != a0 and S[s, a0] == 1:
                sw_live += 1
                pw = np.flatnonzero(S[s:, a0] == 2)
                if len(pw) and t[s + pw[0]] - t[s] <= 2.0: sw_pop2 += 1
                else: far.append(np.linalg.norm(B[s, a0, :3] - R[s, :3]))
    lead = np.array(lead)
    print(f"{kind}: pops {n_pop}; target at the step before the pop != popped balloon: {n_away} (never targeted: {n_never}); "
          f"lead median {np.median(lead) if len(lead) else np.nan:.2f} IQR {np.percentile(lead,25) if len(lead) else np.nan:.2f}-{np.percentile(lead,75) if len(lead) else np.nan:.2f}; "
          f"switches away from live target {sw_live}, popped within 2 s {sw_pop2}, others dist median {np.median(far) if far else np.nan:.1f} m")

"""Mode 3: terminal-velocity shaping (ZEM/ZEV) toward the next balloon, strict order.
a = 6 ZEM / t^2 - 2 ZEV / t, ZEV = v_f* - v, v_f* = (1-kappa) v_f,nat + kappa (v_next_balloon + s * unit(b_next(t) - b_cur(t))),
s = |v_f,nat - v_next_balloon| (same closing speed as the unshaped arrival), t = stretch * t_go (as the baseline).
Only used while the command stays within thrust (|saturated a| not clipped by more than CLIP) and t_go > T_MINZ; else N=2.
Usage: chain_sim3.py KAPPA [TAG]"""
import json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import chain_sim as C  # noqa: E402

T_MINZ = 0.4


class ZEVTeacher(C.ChainTeacher):
    def _guidance(self, p, v, bpos, bvel, t_go):
        a_cur = C.rt.Teacher._guidance(self, p, v, bpos, bvel, t_go)
        if self._obs is None or self.target is None:
            return a_cur
        self.n_steps += 1
        nb = self.next_after(self._obs, self.target)
        t = max(t_go * self.stretch, self.t_go_min)
        if nb is None or t < T_MINZ or (self.coast and t > self.burn_left):
            return a_cur
        bn = self._obs["balloon_states"][nb].astype(float)
        if self.bwind or self.aim_prof is not None:  # same balloon-drift aim as the baseline guidance
            bpos = bpos + self._bdrift(bpos[2], bvel[2], t, self.aim_prof)
        zem = bpos - p + (bvel - v) * t
        vf_nat = v + 2 * zem / t  # N = 2 arrival velocity (constant net accel)
        rc, rn = bpos + bvel * t, bn[:3] + bn[3:] * t
        s = np.linalg.norm(vf_nat - bn[3:])
        vf_star = bn[3:] + s * C.np.asarray((rn - rc) / max(np.linalg.norm(rn - rc), 1e-6))
        vf = (1 - self.w_turn) * vf_nat + self.w_turn * vf_star
        a = 6 * zem / t ** 2 - 2 * (vf - v) / t
        # feasibility: keep the shaped command only if it fits the thrust (after gravity) without clipping
        if np.linalg.norm(a - C.G) > self.a_thrust - 0.3:
            return a_cur
        self.n_chain += 1
        return a


def run(name, kappa):
    C.ChainTeacher = ZEVTeacher  # run() builds C.ChainTeacher
    return C.run(name, 3, 1.0, kappa)


if __name__ == "__main__":
    import multiprocessing as mp
    kappa = float(sys.argv[1]); tag = sys.argv[2] if len(sys.argv) > 2 else f"m3_k{kappa}"
    names = os.environ.get("NAMES", " ".join(C.NAMES)).split()
    with mp.get_context("spawn").Pool(4) as p:
        res = p.starmap(run, [(n, kappa) for n in names])
    json.dump(res, open(Path(__file__).parent / f"chain_{tag}.json", "w"))
    tot, gaps = 0, []
    for r in res:
        P = [tp for tp, _ in r["pops"] if tp <= r["launch"] + 30]
        tot += len(r["pops"]); gaps += list(np.diff(P))
        print(f"{r['name']:14s} pops {len(r['pops']):2d}  shaped steps {r['n_chain']}/{r['n_steps']}")
    print(f"{tag}: total pops {tot}; median hop {np.median(gaps):.2f} s mean {np.mean(gaps):.2f} s (n={len(gaps)})")

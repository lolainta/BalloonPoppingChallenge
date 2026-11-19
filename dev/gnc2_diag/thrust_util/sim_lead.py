"""Strict-order test of 'terminal lead-out': in the last T_LEAD s of a hop, thrust the unused magnitude toward the NEXT
target's needed acceleration (next = their next still-alive balloon after the current one), limited so the extra
displacement before the hit stays <= DELTA m (k <= 2 DELTA / t_go^2). Otherwise identical to sim_variants 'base'.

.venv/bin/python dev/gnc2_diag/thrust_util/sim_lead.py lead10 lead05 ...   (4 processes)
"""
import contextlib, io, json, os, sys
from pathlib import Path

os.environ["SLACK"] = "inf"
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sim_variants as sv  # noqa: E402

LEAD = {"lead10": (1.0, 0.7, 0), "lead05": (0.5, 0.7, 0), "lead10f": (1.0, 0.7, 1), "lead15": (1.5, 1.0, 0)}  # (T_LEAD, DELTA, full)


def make_agent(given, info, variant):
    from dev import replay_teacher as rt
    T_LEAD, DELTA, FULL = LEAD[variant]
    rt.FIX_S = None

    class T(rt.Teacher):
        def _select_target(self, p, v, obs):
            b, tg = super()._select_target(p, v, obs)
            self._obs, self._tgo_now, self._pv = obs, (np.nan if tg is None else tg * self.stretch), (p, v)
            return b, tg

        def _next_target(self, obs):
            st = obs["balloon_status"][:, 0]
            seen = False
            for _, b in self.info["pops"]:
                b = int(b)
                if b == self.target:
                    seen = True
                    continue
                if seen and st[b] == 1:
                    return b
            return None

        def _spend_excess(self, f, v):
            tgo = getattr(self, "_tgo_now", np.nan)
            if self.target is None or not np.isfinite(tgo) or tgo > T_LEAD:
                return f
            nb = self._next_target(self._obs)
            if nb is None:
                return f
            p, _ = self._pv
            b = self._obs["balloon_states"][nb]
            # the next leg's needed acceleration, measured from the current target's arrival point/time
            cur = self._obs["balloon_states"][self.target]
            p_arr, t_arr = cur[:3] + cur[3:] * tgo, tgo
            t2 = self._reach_time(p_arr, v, b[None, :3] + b[None, 3:] * t_arr, b[None, 3:])[0]
            if not np.isfinite(t2):
                return f
            a_next = 2.0 * (b[:3] + b[3:] * (t_arr + t2) - p_arr - v * t2) / t2**2 - np.array([0.0, 0.0, -9.81])
            d = a_next / max(np.linalg.norm(a_next), 1e-9)
            bb, c = f @ d, f @ f - self.a_thrust**2
            k = max(-bb + np.sqrt(max(bb * bb - c, 0.0)), 0.0)  # |f + k d| = a_thrust
            k = min(k, 2.0 * DELTA / max(tgo, 0.05) ** 2)
            f2 = f + k * d
            if FULL:  # then fill any remaining magnitude along f2
                f2 = f2 * max(1.0, self.a_thrust / max(np.linalg.norm(f2), 1e-9))
            return f2

    return T(given, info, policy_path=sv.CK, sl_path=sv.CK.replace("rl3_policy", "rl2_sl"))


def run(name, variant):
    sv.make_agent = make_agent  # reuse the logger with this factory
    return sv.run(name, variant)


if __name__ == "__main__":
    import multiprocessing as mp
    variants = sys.argv[1:] or ["lead10"]
    jobs = [(n, v) for v in variants for n in sv.NAMES]
    with mp.get_context("spawn").Pool(4) as p:
        res = p.starmap(run, jobs)
    for v in variants:
        rs = [r for r in res if r["variant"] == v]
        json.dump(rs, open(sv.OUT / f"summary_{v}.json", "w"), indent=1, default=float)
        print(v, sum(r["pops"] for r in rs))

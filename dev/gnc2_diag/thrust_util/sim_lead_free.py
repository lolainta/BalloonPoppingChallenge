"""Free-mode (our policy picks launch / targets / speed levels) test of the terminal lead-out, with the NEXT target
PREDICTED (no replay knowledge): the alive balloon (not the current one) with the least full-thrust reach time from the
current target's arrival point/time. variants: freebase (no lead), freelead10 (T_LEAD 1.0 s, DELTA 0.7 m).

.venv/bin/python dev/gnc2_diag/thrust_util/sim_lead_free.py freebase freelead10   (4 processes)
"""
import contextlib, io, json, os, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sim_variants as sv  # noqa: E402

LEAD = {"freebase": None, "freelead10": (1.0, 0.7), "freelead05": (0.5, 0.7)}


def make_agent(given, variant):
    from BalloonPoppingGymEnv.agents.rl3_agent import RL3Agent
    cfg = LEAD[variant]

    class A(RL3Agent):
        def _select_target(self, p, v, obs):
            b, tg = super()._select_target(p, v, obs)
            self._obs, self._tgo_now, self._pv = obs, (np.nan if tg is None else tg * self.stretch), (p, v)
            return b, tg

        def _spend_excess(self, f, v):
            if cfg is None:
                return f
            T_LEAD, DELTA = cfg
            tgo = getattr(self, "_tgo_now", np.nan)
            if self.target is None or not np.isfinite(tgo) or tgo > T_LEAD:
                return f
            obs = self._obs
            st, bs = obs["balloon_status"][:, 0], obs["balloon_states"]
            alive = np.flatnonzero((st == 1) & (np.arange(len(st)) != self.target))
            if alive.size == 0:
                return f
            cur = bs[self.target]
            p_arr = cur[:3] + cur[3:] * tgo
            t2 = self._reach_time(p_arr, v, bs[alive, :3] + bs[alive, 3:] * tgo, bs[alive, 3:])
            j = int(np.argmin(t2))
            if not np.isfinite(t2[j]):
                return f
            b, t2 = bs[alive[j]], t2[j]
            a_next = 2.0 * (b[:3] + b[3:] * (tgo + t2) - p_arr - v * t2) / t2**2 - np.array([0.0, 0.0, -9.81])
            d = a_next / max(np.linalg.norm(a_next), 1e-9)
            bb, c = f @ d, f @ f - self.a_thrust**2
            k = max(-bb + np.sqrt(max(bb * bb - c, 0.0)), 0.0)
            k = min(k, 2.0 * DELTA / max(tgo, 0.05) ** 2)
            return f + k * d

    return A(given, policy_path=sv.CK, sl_path=sv.CK.replace("rl3_policy", "rl2_sl"))


def run(name, variant):
    import warnings; warnings.filterwarnings("ignore")
    from dev import replay_teacher as rt
    from dev import rl2_env
    info = rt.replay_info(str(sv.ROOT.parent / f"lb_replays/{name}.json"))
    params, given = rl2_env.load_scenario_parameters(1)
    env = rl2_env.BalloonPoppingEnv(render_mode=None, parameters=params)
    with contextlib.redirect_stdout(io.StringIO()):
        obs, _ = env.reset(seed=info["seed"])
    ag = make_agent(given, variant)
    L = {k: [] for k in "t R S A F TGT STRETCH".split()}
    Bl = []
    done = False
    while not done:
        a = ag.get_action(obs)
        obs, r, term, trunc, inf = env.step(a)
        done = term or trunc
        dbg = ag.debug or {}
        L["t"].append(float(obs["simulation_time"])); L["R"].append(np.asarray(inf["rocket_states"], float))
        Bl.append(obs["balloon_states"].astype(np.float32)); L["S"].append(obs["balloon_status"][:, 0].astype(np.int8))
        L["A"].append([*np.asarray(a["tvc"], float), float(a["roll"]), float(a["throttle"])])
        L["F"].append(np.asarray(dbg.get("f", [np.nan] * 3), float)); L["TGT"].append(-1 if dbg.get("target") is None else int(dbg["target"]))
        L["STRETCH"].append(float(ag.stretch))
    S = np.array(L["S"]); t = np.array(L["t"])
    pops = sorted((t[int(np.argmax(S[:, b] == 2))], b) for b in range(S.shape[1]) if (S[:, b] == 2).any())
    R = np.array(L["R"]); live = ~np.isnan(R[:, 0])
    sv.OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(sv.OUT / f"{variant}_{name}.npz", **{k: np.array(v) for k, v in L.items()}, B=np.array(Bl),
                        launch=float(t[int(np.argmax(live))]), pops=np.array(pops, float), seed=info["seed"])
    return dict(name=name, variant=variant, pops=len(pops))


if __name__ == "__main__":
    import multiprocessing as mp
    variants = sys.argv[1:] or ["freebase", "freelead10"]
    jobs = [(n, v) for v in variants for n in sv.NAMES]
    with mp.get_context("spawn").Pool(4) as p:
        res = p.starmap(run, jobs)
    for v in variants:
        rs = [r for r in res if r["variant"] == v]
        json.dump(rs, open(sv.OUT / f"summary_{v}.json", "w"), indent=1, default=float)
        print(v, sum(r["pops"] for r in rs), [r["pops"] for r in rs])

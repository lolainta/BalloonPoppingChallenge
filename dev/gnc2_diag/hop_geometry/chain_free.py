"""Free mode (our policy picks launch / targets / speed levels) + early handover to a PREDICTED next target.
nextsrc 1: the policy's own pick with the current target masked (one extra forward per decision)
nextsrc 2: GNC: the alive balloon with the min reach time from the current state (excluding the current target)
commit 1 : after a pop, if the predicted next is alive the agent targets it (the policy still picks the speed level)
mode 0 = baseline. Usage: chain_free.py MODE NEXTSRC COMMIT R_OK TAG [SEEDS as a-b]"""
import contextlib, io, json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import chain_sim as C  # noqa: E402
from BalloonPoppingGymEnv.agents.rl3_agent import RL3Agent, G_DIM  # noqa: E402
from BalloonPoppingGymEnv.agents.rl2_agent import F, N_MAX  # noqa: E402


class FreeChain(RL3Agent):
    def __init__(self, given, mode=0, nextsrc=1, commit=0, r_ok=1.0, **kw):
        super().__init__(given, **kw)
        self.mode, self.nextsrc, self.commit, self.r_ok = mode, nextsrc, commit, r_ok
        self.next_b, self._obs, self.n_chain, self.n_steps, self.n_commit, self.n_pred_ok, self.n_pred = None, None, 0, 0, 0, 0, 0
        self._pending = None

    def decide(self, obs, action):
        st = obs["balloon_status"][:, 0]
        popped_now = self.cmd is not None and int((st == 2).sum()) > self.n_popped
        if popped_now and self._pending is not None:  # how often was the prediction the policy's pick?
            self.n_pred += 1; self.n_pred_ok += int(action[0]) == self._pending
        if self.mode and self.commit and popped_now and self.next_b is not None and st[self.next_b] == 1:
            action = (self.next_b, action[1]) + tuple(action[2:]); self.n_commit += 1
        super().decide(obs, action)
        self.next_b = self._predict_next(obs) if self.cmd is not None else None
        self._pending = self.next_b

    def _predict_next(self, obs):
        cur = self.cmd[0]; st = obs["balloon_status"][:, 0]
        if self.nextsrc == 1:
            x = self.features(obs)
            x[G_DIM + cur * F + 14] = 0.0  # mask the current target
            tl = self.forward(x)[0]
            if (x[G_DIM:].reshape(N_MAX, F)[:, 14] > 0.5).sum() == 0:
                return None
            return int(np.argmax(tl))
        alive = np.flatnonzero(st == 1); alive = alive[alive != cur]
        if not alive.size:
            return None
        bs = obs["balloon_states"]
        t = self._reach_time(self.p_hat, self.v_hat, bs[alive, :3], bs[alive, 3:])
        return int(alive[np.argmin(t)]) if np.isfinite(t).any() else None

    def _gnc_action(self, obs):
        self._obs = obs
        return super()._gnc_action(obs)

    def _guidance(self, p, v, bpos, bvel, t_go):
        a_cur = super()._guidance(p, v, bpos, bvel, t_go)
        if not self.mode or self._obs is None or self.target is None:
            return a_cur
        self.n_steps += 1
        nb = self.next_b
        if nb is None or nb == self.target or self._obs["balloon_status"][nb, 0] != 1:
            return a_cur
        bn = self._obs["balloon_states"][nb].astype(float)
        tn = self._reach_time(p, v, bn[None, :3], bn[None, 3:])[0]
        if not np.isfinite(tn):
            return a_cur
        a_next = super()._guidance(p, v, bn[:3], bn[3:], tn)
        hor = 1.5 * max(t_go * self.stretch, self.t_go_min) + 0.5
        net = self._saturate(a_next) + C.G
        rel = (bpos - p)[None] + (bvel - v)[None] * C.TH[:, None] - 0.5 * net[None] * C.TH[:, None] ** 2
        if ((np.linalg.norm(rel, axis=1) < self.r_ok) & (C.TH <= hor)).any():
            self.n_chain += 1
            return a_next
        return a_cur


def run(seed, mode, nextsrc, commit, r_ok):
    import warnings; warnings.filterwarnings("ignore")
    params, given = C.rl2_env.load_scenario_parameters(1)
    env = C.rl2_env.BalloonPoppingEnv(render_mode=None, parameters=params)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            obs, _ = env.reset(seed=seed)
    except Exception as e:  # some seeds crash in reset
        return dict(seed=seed, err=str(e)[:80])
    ag = FreeChain(given, mode=mode, nextsrc=nextsrc, commit=commit, r_ok=r_ok, policy_path=C.CK, sl_path=C.CK.replace("rl3_policy", "rl2_sl"))
    done, S, T, launch = False, [], [], None
    while not done:
        obs, r, term, trunc, inf = env.step(ag.get_action(obs))
        done = term or trunc
        t = float(obs["simulation_time"]); T.append(t); S.append(obs["balloon_status"][:, 0].copy())
        if launch is None and not np.isnan(np.asarray(inf["rocket_states"], float)[0]):
            launch = t
    S = np.array(S); T = np.array(T)
    pops = sorted((float(T[int(np.argmax(S[:, b] == 2))]), int(b)) for b in range(S.shape[1]) if (S[:, b] == 2).any())
    return dict(seed=seed, pops=pops, launch=launch, n_chain=ag.n_chain, n_steps=ag.n_steps, n_commit=ag.n_commit,
                pred_ok=ag.n_pred_ok, pred_n=ag.n_pred)


if __name__ == "__main__":
    import multiprocessing as mp
    mode, nextsrc, commit, r_ok, tag = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
    lo, hi = (int(x) for x in (sys.argv[6] if len(sys.argv) > 6 else "20000-20019").split("-"))
    with mp.get_context("spawn").Pool(4) as p:
        res = p.starmap(run, [(s, mode, nextsrc, commit, r_ok) for s in range(lo, hi + 1)])
    json.dump(res, open(Path(__file__).parent / f"free_{tag}.json", "w"))
    ok = [r for r in res if "pops" in r]
    gaps = []
    for r in ok:
        P = [tp for tp, _ in r["pops"] if tp <= r["launch"] + 30]; gaps += list(np.diff(P))
    pn = sum(r["pred_n"] for r in ok)
    print(f"{tag}: seeds {len(ok)} mean pops {np.mean([len(r['pops']) for r in ok]):.2f} per seed {[len(r['pops']) for r in ok]} "
          f"median hop {np.median(gaps):.2f} mean {np.mean(gaps):.2f}  next-pred agreement {sum(r['pred_ok'] for r in ok)}/{pn}")

"""Free mode with hit-constrained blending that fits how our policy already hands over (it switches to the next balloon
~0.58 s before popping the current one).
  guard (g>=1): when the policy switches from A to B while A is alive, A becomes "committed": the GNC flies the blend
                a = (1-l) a_A + l a_B, l in 0..1, that still passes A within R_OK (constant-accel prediction) and minimises
                t_hit(A) + t_reach(B | arrival); A is dropped when popped, unreachable under every blend, or after 2.5 s.
  early (g=2) : before any switch, the same blend with must-hit = policy target, next = the policy's pick with its
                current target masked (computed at each decision).
Usage: chain_free2.py G R_OK TAG [SEEDS a-b]"""
import contextlib, io, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import chain_sim as C  # noqa: E402
from BalloonPoppingGymEnv.agents.rl3_agent import RL3Agent, G_DIM  # noqa: E402
from BalloonPoppingGymEnv.agents.rl2_agent import F  # noqa: E402


class FreeChain2(RL3Agent):
    def __init__(self, given, g=1, r_ok=1.0, **kw):
        super().__init__(given, **kw)
        self.g, self.r_ok = g, r_ok
        self.committed, self.t_commit, self.pred, self._obs = None, 0.0, None, None
        self.n_guard = self.n_early = self.n_drop = self.n_commit = 0

    def decide(self, obs, action):
        st = obs["balloon_status"][:, 0]; t = float(obs["simulation_time"])
        old = self.cmd[0] if self.cmd is not None else None
        super().decide(obs, action)
        new = self.cmd[0] if self.cmd is not None else None
        if self.g >= 1 and old is not None and new is not None and old != new and st[old] == 1:
            self.committed, self.t_commit = old, t; self.n_commit += 1
        if self.g >= 2 and new is not None:
            x = self.features(obs); x[G_DIM + new * F + 14] = 0.0
            tl = self.forward(x)[0]; self.pred = int(np.argmax(tl)) if tl.max() > -1e8 else None

    def _gnc_action(self, obs):
        self._obs = obs
        return super()._gnc_action(obs)

    def _t_hit(self, p, v, net, bp, bv, horizon):
        rel = (bp - p)[None] + (bv - v)[None] * C.TH[:, None] - 0.5 * net[None] * C.TH[:, None] ** 2
        ok = (np.linalg.norm(rel, axis=1) < self.r_ok) & (C.TH <= horizon)
        return C.TH[int(np.argmax(ok))] if ok.any() else np.inf

    def _blend(self, p, v, a_hit, hp, hv, a_nxt, np_, nv, hor):
        """min over l of t_hit(must-hit) + t_reach(next | arrival); None if no blend passes the must-hit balloon."""
        best, a_best = np.inf, None
        for lam in C.LAM:
            a = (1 - lam) * a_hit + lam * a_nxt
            net = self._saturate(a) + C.G
            th = self._t_hit(p, v, net, hp, hv, hor)
            if not np.isfinite(th):
                continue
            ph, vh = p + v * th + 0.5 * net * th ** 2, v + net * th
            t2 = self._reach_time(ph, vh, (np_ + nv * th)[None], nv[None])[0]
            c = th + (t2 if np.isfinite(t2) else 20.0)
            if c < best - 1e-9:
                best, a_best = c, a
        return a_best

    def _guidance(self, p, v, bpos, bvel, t_go):
        a_tgt = super()._guidance(p, v, bpos, bvel, t_go)
        obs = self._obs
        if not self.g or obs is None or self.target is None:
            return a_tgt
        st, bs, t = obs["balloon_status"][:, 0], obs["balloon_states"], float(obs["simulation_time"])
        A = self.committed
        if A is not None and (st[A] != 1 or t - self.t_commit > 2.5 or A == self.target):
            self.committed = A = None
        if A is not None:  # must hit A, next = policy target
            ba = bs[A].astype(float)
            ta = self._reach_time(p, v, ba[None, :3], ba[None, 3:])[0]
            a_A = super()._guidance(p, v, ba[:3], ba[3:], ta if np.isfinite(ta) else 1.0)
            a = self._blend(p, v, a_A, ba[:3], ba[3:], a_tgt, np.asarray(bpos, float), np.asarray(bvel, float), 2.5)
            if a is None:
                self.committed = None; self.n_drop += 1
                return a_tgt
            self.n_guard += 1
            return a
        if self.g >= 2 and self.pred is not None and self.pred != self.target and st[self.pred] == 1:
            bn = bs[self.pred].astype(float)
            tn = self._reach_time(p, v, bn[None, :3], bn[None, 3:])[0]
            if np.isfinite(tn):
                a_n = super()._guidance(p, v, bn[:3], bn[3:], tn)
                hor = 1.5 * max(t_go * self.stretch, self.t_go_min) + 0.5
                a = self._blend(p, v, a_tgt, np.asarray(bpos, float), np.asarray(bvel, float), a_n, bn[:3], bn[3:], hor)
                if a is not None:
                    self.n_early += 1
                    return a
        return a_tgt


def run(seed, g, r_ok):
    import warnings; warnings.filterwarnings("ignore")
    params, given = C.rl2_env.load_scenario_parameters(1)
    env = C.rl2_env.BalloonPoppingEnv(render_mode=None, parameters=params)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            obs, _ = env.reset(seed=seed)
    except Exception as e:
        return dict(seed=seed, err=str(e)[:80])
    ag = FreeChain2(given, g=g, r_ok=r_ok, policy_path=C.CK, sl_path=C.CK.replace("rl3_policy", "rl2_sl"))
    done, S, T, launch = False, [], [], None
    while not done:
        obs, r, term, trunc, inf = env.step(ag.get_action(obs)); done = term or trunc
        t = float(obs["simulation_time"]); T.append(t); S.append(obs["balloon_status"][:, 0].copy())
        if launch is None and not np.isnan(np.asarray(inf["rocket_states"], float)[0]):
            launch = t
    S = np.array(S); T = np.array(T)
    pops = sorted((float(T[int(np.argmax(S[:, b] == 2))]), int(b)) for b in range(S.shape[1]) if (S[:, b] == 2).any())
    return dict(seed=seed, pops=pops, launch=launch, guard=ag.n_guard, early=ag.n_early, drop=ag.n_drop, commit=ag.n_commit)


if __name__ == "__main__":
    import multiprocessing as mp
    g, r_ok, tag = int(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
    lo, hi = (int(x) for x in (sys.argv[4] if len(sys.argv) > 4 else "20000-20019").split("-"))
    with mp.get_context("spawn").Pool(4) as p:
        res = p.starmap(run, [(s, g, r_ok) for s in range(lo, hi + 1)])
    json.dump(res, open(Path(__file__).parent / f"free_{tag}.json", "w"))
    ok = [r for r in res if "pops" in r]; gaps = []
    for r in ok:
        P = [tp for tp, _ in r["pops"] if tp <= r["launch"] + 30]; gaps += list(np.diff(P))
    print(f"{tag}: seeds {len(ok)} mean pops {np.mean([len(r['pops']) for r in ok]):.2f} per seed {[len(r['pops']) for r in ok]} "
          f"median hop {np.median(gaps):.2f} mean {np.mean(gaps):.2f} | commits {sum(r['commit'] for r in ok)} drops {sum(r['drop'] for r in ok)} "
          f"guard steps {sum(r['guard'] for r in ok)} early steps {sum(r['early'] for r in ok)}")

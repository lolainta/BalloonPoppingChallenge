"""RL guidance on top of GNCAgent: a policy picks (target, speed level); GNC's ZEM guidance,
navigation and attitude control fly it.

Decision points: first burning step, then every DECIDE steps, or right after a pop, or when the
target is gone. Action = (candidate index among the K balloons with the lowest analytic reach time,
stretch level: t_go multiplier, lower = faster). Policy weights: rl_policy.npz, trained offline by
dev/train_ppo.py; at run time the agent only uses observations and given parameters.
"""

from pathlib import Path

import numpy as np

from BalloonPoppingGymEnv.agents.gnc_agent import GNCAgent

K = 16
STRETCH = (0.6, 0.8, 1.0, 1.3)
DECIDE = 50  # steps (0.5 s at dt 0.01)
F = 10  # features per candidate
N_OBS = 6 + F * K


class RLAgent(GNCAgent):
    def __init__(self, given_parameters, **kw):
        super().__init__(given_parameters, **kw)
        self.cmd = None  # (balloon index, stretch)
        self.cand = None
        self.since = 0
        self.n_popped = 0
        path = Path(kw.get("policy_path", Path(__file__).with_name("rl_policy.npz")))
        self.pi = None
        if kw.get("use_policy", 1) and path.exists():
            d = np.load(path)
            self.pi = [(d[f"W{i}"], d[f"b{i}"]) for i in range(int(d["n_layers"]))]

    # ------------------------------------------------------------------ policy interface
    def needs_decision(self, obs):
        st = obs["balloon_status"][:, 0]
        return (
            self.cmd is None
            or self.since >= DECIDE
            or int((st == 2).sum()) > self.n_popped
            or st[self.cmd[0]] != 1
        )

    def features(self, obs):
        """(N_OBS,) observation vector; also stores the candidate list used to decode actions."""
        p, v = self.p_hat, self.v_hat
        st, bs = obs["balloon_status"][:, 0], obs["balloon_states"]
        alive = np.flatnonzero(st == 1)
        x = np.zeros(N_OBS, np.float32)
        t = float(obs["simulation_time"])
        x[:6] = [*(v / 30.0), (p[2] - self.elev) / 100.0, (self.launched_at + self.burn_time - t) / 30.0, self.a_thrust / 15.0]
        self.cand = np.full(K, -1)
        if alive.size:
            bp, bv = bs[alive, :3], bs[alive, 3:]
            t1 = self._reach_time(p, v, bp, bv)
            dist = np.linalg.norm(bp - p, axis=1)
            order = np.lexsort((dist, np.where(np.isfinite(t1), t1, 1e3)))[:K]
            cur = self.cmd[0] if self.cmd is not None else -1
            if cur in alive and cur not in alive[order]:  # always show the current target (last slot)
                order[-1 if order.size == K else order.size:] = np.flatnonzero(alive == cur)[0]
                if order.size < K and cur not in alive[order]:
                    order = np.append(order, np.flatnonzero(alive == cur)[0])
            for j, i in enumerate(order):
                rel, relv = bp[i] - p, bv[i] - v
                tt = t1[i] if np.isfinite(t1[i]) else 20.0
                x[6 + F * j : 6 + F * (j + 1)] = [*(rel / 50.0), *(relv / 20.0), tt / 10.0, (bp[i, 2] - self.elev) / 100.0,
                                                  1.0, float(alive[i] == cur)]
                self.cand[j] = alive[i]
        return x

    def decide(self, obs, action):
        i, s = int(action[0]), int(action[1])
        valid = self.cand[self.cand >= 0]
        if valid.size == 0:
            self.cmd = None
        else:
            self.cmd = (int(self.cand[i]) if self.cand[i] >= 0 else int(valid[0]), STRETCH[s])
        self.since = 0
        self.n_popped = int((obs["balloon_status"][:, 0] == 2).sum())

    def _policy(self, x):
        a = x
        for W, b in self.pi[:-1]:
            a = np.tanh(a @ W + b)
        W, b = self.pi[-1]
        logits = a @ W + b
        return int(np.argmax(logits[:K])), int(np.argmax(logits[K:]))

    # ------------------------------------------------------------------ GNC hooks
    def _select_target(self, p, v, obs):
        if self.cmd is None or obs["balloon_status"][self.cmd[0], 0] != 1:
            return None, None
        tgt, self.stretch = self.cmd
        b = obs["balloon_states"][tgt]
        t1 = self._reach_time(p, v, b[None, :3], b[None, 3:])[0]
        return tgt, (t1 if np.isfinite(t1) else self.t_grid[-1])

    def get_action(self, obs):
        t = float(obs["simulation_time"])
        if self.pi is not None and self.p_hat is not None and self._burning(t) and self.needs_decision(obs):
            self.decide(obs, self._policy(self.features(obs)))
        self.since += 1
        return super().get_action(obs)

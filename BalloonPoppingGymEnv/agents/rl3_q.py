"""RL3Q = RL3Agent + one-step policy improvement with the learned critic on close calls (evaluation only).

At a post-launch decision where the policy's argmax balloon j1 is not the held target, the critic is queried on the REAL
current state twice, once with j1 and once with the policy's second choice j2 flagged as the current target (token column 8;
the stretch of self.cmd does not enter the features). That is what the policy sees at its next row after committing to that
balloon, so V(s, current=j) ~ Q(s, j). On a close call (tl[j1] - tl[j2] < qv_gap) with V_j2 - V_j1 > qv_delta it flies j2
instead (at most qv_max times per episode) and holds it while alive for at most qv_hold sim seconds.

Features vs self.cmd: only cmd[0] enters, as the "is current target" token column 8 (rl2_agent.features); the SL reach model
reads column 8 too (rl2_sl.IN_TOK), so the flagged balloon's t_hat / p_hat (columns 9, 10) and the chain columns 15-17 built
from them change with it. cmd[1] (the stretch) is used only by the GNC (_select_target), never by the features.

Critic units (dev/async_ppo.py, dev/rl2_env.py, dev/rl3_env.py): raw discounted pops, no reward or value scaling. Reward of a
row = balloons popped until the next row (after burnout the coast's pops all go to the last row); value target = GAE(lambda)
return A + V (VF 0.5 MSE, only advantages are normalised). With the RL3_FAST policies (fast.k in the npz: rows every fast_k sim
steps between decisions) the discount is per sim time: GAMMA ** (dt / 0.5 s), GAMMA = 0.99 (RL3_GAMMA, unset on the s40 lines),
lambda 0.95 ** (dt / 0.5 s); RL3_TCOST (reward - c per sim s) is off on s40k/l/m/n. So V = E[sum over later pops of
0.99 ** ((t_row - t) / 0.5 s)] (~0.98 per s: a pop 3 s out counts 0.94), and V2 - V1 is a difference in near-term pops.
SIL's value term (0.01 * 0.5 * max(R - V, 0)^2 on the best episodes) biases V slightly up. qv_delta 0.2 ~ 0.2 pops
(0.1-0.3 pops: qv_delta 0.1-0.3).

kwargs (all off by default: without qv_gap / qv_log RL3Q is RL3Agent, action for action):
  qv_gap   (0)    eligible when tl[j1] - tl[j2] < qv_gap (0: never)
  qv_delta (0.2)  override when V_j2 - V_j1 > qv_delta (critic units = discounted pops)
  qv_max   (1)    overrides per episode
  qv_hold  (2.0)  sim s an overridden balloon is held (while alive)
  qv_shadow (0)   1: score and log, never change the action
  qv_wall  (300)  wall s since construction after which scoring stops for the episode
  qv_log   (none) CSV prefix: rows appended to f"{prefix}.{pid}.csv" (kind dec / pop / wall; dev/qv_report.py)
  qv_seed, qv_cfg: written into the rows (dev/sweep_gnc.py passes the episode's seed and config index)
"""

import os
import time

import numpy as np

from BalloonPoppingGymEnv.agents.rl2_agent import STRETCH
from BalloonPoppingGymEnv.agents.rl3_agent import RL3Agent

COLS = ("kind", "run", "seed", "cfg", "t", "t_bo", "held", "j1", "j2", "gap", "V1", "V2", "eligible", "override", "pops")


class RL3Q(RL3Agent):
    def __init__(self, given_parameters, **kw):
        super().__init__(given_parameters, **kw)
        self.qv_gap, self.qv_delta = float(kw.get("qv_gap", 0.0)), float(kw.get("qv_delta", 0.2))
        self.qv_max, self.qv_hold = int(kw.get("qv_max", 1)), float(kw.get("qv_hold", 2.0))
        self.qv_shadow, self.qv_wall = int(kw.get("qv_shadow", 0)), float(kw.get("qv_wall", 300.0))
        self.qv_log = kw.get("qv_log") or None
        self.qv_seed, self.qv_cfg = int(kw.get("qv_seed", -1)), int(kw.get("qv_cfg", -1))
        self.qv_on = self.qv_gap > 0 or self.qv_log is not None
        if self.qv_log and os.path.dirname(str(self.qv_log)):
            os.makedirs(os.path.dirname(str(self.qv_log)), exist_ok=True)
        if self.qv_on and (self.search_k > 0 or self.dp or self.dp_shadow):
            raise NotImplementedError("qv_*: the plain policy argmax only (no search_k / DP)")
        self.qv_t0, self.qv_run = time.monotonic(), f"{os.getpid()}-{time.time_ns()}"
        self.qv_n, self.qv_hold_j, self.qv_hold_until, self.qv_walled, self.qv_npop = 0, None, -np.inf, False, 0

    def _qv_row(self, **r):
        fn = f"{self.qv_log}.{os.getpid()}.csv"
        new = not os.path.exists(fn) or os.path.getsize(fn) == 0
        r = {"run": self.qv_run, "seed": self.qv_seed, "cfg": self.qv_cfg, **r}
        with open(fn, "a") as f:
            if new:
                f.write(",".join(COLS) + "\n")
            f.write(",".join("" if r.get(c) is None else (f"{r[c]:.6g}" if isinstance(r[c], float) else str(r[c])) for c in COLS) + "\n")

    def _qv_snap(self):
        d = dict(self.__dict__); d["opt_n"] = dict(self.opt_n)  # (_cnt counts in place)
        return d

    def _qv_load(self, d):
        self.__dict__.clear(); self.__dict__.update(d); self.opt_n = dict(d["opt_n"])

    def cond(self, i, sl, tl=None):
        if self.qv_on:  # choose()'s (near_k-masked) target logits and unconditioned stretch logits
            self.qv_tl, self.qv_sl = tl, sl
        return super().cond(i, sl, tl)

    def choose(self, obs, tgt=None):
        if tgt is not None or not self.qv_on:
            return super().choose(obs, tgt)
        st, t = obs["balloon_status"][:, 0], float(obs["simulation_time"])
        if self.qv_hold_j is not None:
            if st[self.qv_hold_j] == 1 and t < self.qv_hold_until:
                return super().choose(obs, tgt=self.qv_hold_j)
            self.qv_hold_j = None
        if self.qv_walled or time.monotonic() - self.qv_t0 > self.qv_wall:
            if not self.qv_walled and self.qv_log:
                self._qv_row(kind="wall", t=t, pops=int((st == 2).sum()))
            self.qv_walled = True
            return super().choose(obs)
        s0 = self._qv_snap()
        a = super().choose(obs)
        j1, tl, sl = int(a[0]), self.qv_tl, self.qv_sl
        held = self.cmd[0] if self.cmd is not None and st[self.cmd[0]] == 1 else None
        j2 = self.second(tl, j1)
        if j1 == held or j2 is None:
            return a
        s1 = self._qv_snap()
        V = []
        for j in (j1, j2):  # the real state with j flagged as the current target (= get_action's redecide pass)
            self._qv_load(s0)
            self.cmd = (j, STRETCH[int(np.argmax(self.cond(j, sl, tl)))])
            V.append(float(self._value(self.features(obs))))
        self._qv_load(s1)  # every attribute back to the normal path's (action, res_mean, res_u, h_tok, j2, caches, counts)
        gap = float(tl[j1] - tl[j2])
        elig = gap < self.qv_gap
        ovr = elig and V[1] - V[0] > self.qv_delta and self.qv_n < self.qv_max
        if ovr:
            self.qv_n += 1
            if not self.qv_shadow:
                self._qv_load(dict(s0, qv_n=self.qv_n))
                a = super().choose(obs, tgt=j2)  # stretch / residual / FAST conditioned on the balloon flown
                self.qv_hold_j, self.qv_hold_until = j2, t + self.qv_hold
                self._cnt("qv_override")
        if self.qv_log:
            self._qv_row(kind="dec", t=t, t_bo=float(self.launched_at + self.burn_time), held=-1 if held is None else int(held),
                         j1=j1, j2=j2, gap=gap, V1=V[0], V2=V[1], eligible=int(elig), override=int(ovr), pops=int((st == 2).sum()))
        return a

    def get_action(self, obs):
        if self.qv_log:  # pops seen at this step (the realized returns of dev/qv_report.py)
            n = int((obs["balloon_status"][:, 0] == 2).sum())
            for _ in range(n - self.qv_npop):
                self._qv_row(kind="pop", t=float(obs["simulation_time"]), pops=n)
            self.qv_npop = n
        return super().get_action(obs)

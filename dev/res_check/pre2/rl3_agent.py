"""RL guidance v3 = RL2 + the policy decides WHEN to launch.

Before launch the agent decides every DECIDE steps from T_MIN: action[2] = 1 launches now, 0 waits
(forced launch at T_MAX). Pre-launch the rocket state is the pad (0, 0, elevation) at rest, burn left =
full burn. Two extra global features: launched flag, time / 60 s. After launch it behaves exactly like RL2.
Weights: rl3_policy.npz (dev/train_ppo2.py --launch). Uses only observations and given parameters.
"""

from pathlib import Path

import numpy as np

from BalloonPoppingGymEnv.agents import rl2_agent as v2
from BalloonPoppingGymEnv.agents.gnc_agent import get_initial_attitude, quat_to_R
from BalloonPoppingGymEnv.agents.rl2_agent import DECIDE, F, N_MAX, STRETCH, RL2Agent

G_DIM = v2.G_DIM + 2
N_OBS = G_DIM + N_MAX * F
# launch attitude choices (inclination, heading): a checkpoint with an "att" head picks one at launch, else vertical
ATT = [(90.0, 0.0)] + [(i, float(h)) for i in (85.0, 75.0) for h in range(0, 360, 45)]
# 2026-09-25 23:30: steeper tilts (the top teams launch at 62-68 deg and skim the ground first) and "toward the balloons
# still below 10 m" (heading None, computed at launch). Appended, so older 17-way checkpoints keep their meaning.
ATT += [(i, float(h)) for i in (70.0, 65.0, 60.0) for h in range(0, 360, 45)] + [(i, None) for i in (70.0, 65.0, 60.0)]
# continuous launch attitude (a checkpoint with an "attc" head, 2026-09-25 23:45): pre-squash u = (u1, u2) ->
# inclination 55 + 35 sigmoid(u1) deg, heading = toward the balloons near the ground + 180 tanh(u2) deg
ELEV_LO, ELEV_SPAN = 55.0, 35.0


def att_cont(u):
    return ELEV_LO + ELEV_SPAN / (1.0 + np.exp(-float(u[0]))), 180.0 * np.tanh(float(u[1]))
RES_LS = (float(np.log(0.05)), float(np.log(0.6)))  # clamp of a "res" head's log std (training)
T_MIN, T_MAX = 5.0, 60.0  # T_MIN: default earliest launch decision; a checkpoint may carry its own "t_min"


class RL3Agent(RL2Agent):
    def __init__(self, given_parameters, **kw):
        kw.setdefault("policy_path", Path(__file__).with_name("rl3_policy.npz"))
        super().__init__(given_parameters, **kw)
        self.t_min = float(kw.get("t_min", T_MIN))
        if self.net is not None and "t_min" in self.net and "t_min" not in kw:  # the launch floor the policy was
            self.t_min = float(self.net["t_min"])  # trained with; an explicit t_min kwarg wins (e.g. not before 25 s)
        for k, v in (self.net or {}).items():  # GNC tunables the policy was trained with ("gnc.k_att", ...);
            if k.startswith("gnc.") and k[4:] not in kw:  # an explicit kwarg still wins (A/B tests)
                setattr(self, k[4:], float(v))
        # learned guidance residual (a checkpoint with a "res" head, 2026-10-01): at each post-launch decision
        # res = res_bound * a_thrust * tanh(u), u = the head's mean (training: a sample), held until the next decision
        self.res_bound = float(kw.get("res_bound", (self.net or {}).get("res.bound", 0.0)))
        self.res_u = None
        self.launch_time = np.inf  # set by the policy (or T_MAX)
        self.pre_since = DECIDE
        # how a deterministic run turns the per-decision launch probabilities into a launch:
        #   "argmax": launch when P(launch) > 0.5 at a decision (training samples, so this can wait far longer)
        #   "median": launch once the chance of having launched by now under the policy, 1 - prod(1 - P), reaches 0.5,
        #             i.e. at the median of the policy's own launch time; still decided online from the observations
        self.launch_rule = kw.get("launch_rule", "argmax")
        self.launch_incl = kw.get("launch_incl", 0.0)  # > 0: override the launch attitude (see decide_launch)
        self.p_wait = 1.0  # prod(1 - P(launch)) over the launch decisions so far

    # ------------------------------------------------------------------ features
    def launched(self):
        return np.isfinite(self.launch_time)

    def needs_launch_decision(self, obs):
        t = float(obs["simulation_time"])
        return not self.launched() and t >= self.t_min and self.pre_since >= DECIDE

    def features(self, obs):
        t = float(obs["simulation_time"])
        pre = self.p_hat is None
        if pre:  # on the pad: fake the post-launch state the RL2 features expect
            saved = (self.p_hat, self.v_hat, self.launched_at)
            self.p_hat, self.v_hat, self.launched_at = np.array([0.0, 0.0, self.elev]), np.zeros(3), t
        x2 = RL2Agent.features(self, obs)
        if pre:
            self.p_hat, self.v_hat, self.launched_at = saved
        x = np.zeros(N_OBS, np.float32)
        x[: v2.G_DIM] = x2[: v2.G_DIM]
        x[v2.G_DIM : G_DIM] = [0.0 if pre else 1.0, t / 60.0]
        x[G_DIM:] = x2[v2.G_DIM :]
        return x

    def _low_heading(self, obs, incl):
        """Launch heading whose tilted body axis points at the airborne balloons below 10 m (their centroid)."""
        bs, st = obs["balloon_states"], obs["balloon_status"][:, 0]
        low = (st == 1) & (bs[:, 2] - self.elev < 10.0)
        if not low.any():
            low = (st == 1) & (bs[:, 2] - self.elev > 0.5)  # none near the ground: the airborne ones
            if not low.any():
                return 0.0
        d = bs[low, :2].mean(0)  # the pad is at the origin
        best, hb = -2.0, 0.0
        for h in range(0, 360, 5):
            z = quat_to_R(np.array(get_initial_attitude(incl, float(h))))[:, 2]
            c = z[:2] @ d / (np.linalg.norm(z[:2]) * np.linalg.norm(d) + 1e-9)
            if c > best:
                best, hb = c, float(h)
        return hb

    def decide_launch(self, obs, launch, att=0):
        t = float(obs["simulation_time"])
        self.att_last = att  # a launch forced at T_MAX between decisions uses the last choice
        if launch or t >= T_MAX:
            self.launch_time = t
            if np.ndim(att) == 1:  # continuous head: (u1, u2)
                incl, off = att_cont(att)
                head = (self._low_heading(obs, incl) + off) % 360.0
            else:
                incl, head = ATT[att]
            if self.launch_incl:  # experiment override (2026-09-25 23:20): this tilt, toward the balloons near the ground
                incl, head = self.launch_incl, None
            self.incl_head = np.array([incl, self._low_heading(obs, incl) if head is None else head])
            self.q = np.array(get_initial_attitude(*self.incl_head))
        self.pre_since = 0

    def set_res(self, u):
        self.res = self.res_bound * self.a_thrust * np.tanh(np.asarray(u, float))

    # ------------------------------------------------------------------ numpy network (mirrors dev/rl3_net.py)
    def forward(self, x):
        """-> (target logits, stretch logits, launch logits (2,), value, attitude logits (len(ATT),) or attitude mean u (2,)).
        A "res" head's mean u (3,) is left in self.res_u."""
        P = self.net
        g, tok = x[:G_DIM], x[G_DIM:].reshape(N_MAX, F)
        relu = lambda z: np.maximum(z, 0)
        ge = relu(g @ P["g.w"] + P["g.b"])
        h = RL2Agent._trunk(self, tok, ge)
        mask = tok[:, 14] > 0.5
        tl = (h @ P["tgt.w"] + P["tgt.b"])[:, 0] + P["prior"][0] * (-tok[:, 6])
        tl = np.where(mask, tl, -1e9)
        pooled = (h * mask[:, None]).sum(0) / max(mask.sum(), 1)
        pg = np.concatenate([pooled, ge])
        sl = pg @ P["str.w"] + P["str.b"]
        ll = pg @ P["lau.w"] + P["lau.b"]
        ll = ll + np.array([0.0, P["lprior"][0] * (x[v2.G_DIM + 1] * 60.0 - 20.0)])
        val = (relu(pg @ P["v0.w"] + P["v0.b"]) @ P["v1.w"] + P["v1.b"])[0]
        if "res.w" in P:
            self.res_u = pg @ P["res.w"] + P["res.b"]
        if "attc.w" in P:  # continuous attitude: the mean of u (2,)
            al = pg @ P["attc.w"] + P["attc.b"]
        else:
            al = pg @ P["att.w"] + P["att.b"] if "att.w" in P else np.eye(len(ATT))[0]  # no head: vertical
        return tl, sl, ll, val, al

    def _value(self, x):
        return self.forward(x)[3]

    def choose(self, obs):
        self.res_mean = None
        a = self._dp_choose(obs) if self.dp == 1 else None  # DP planner (RL2Agent._dp_choose)
        if a is not None:
            return a
        x = self.features(obs)
        tl, sl, *_ = self.forward(x)
        self.res_mean = self.res_u  # before _search's look-ahead forwards overwrite it
        s = int(np.argmax(sl))
        a = (self._search(obs, x, tl, s) if self.search_k > 0 else int(np.argmax(tl))), s
        return self._dp_choose(obs, tl, a) if self.dp or self.dp_shadow else a

    def get_action(self, obs):
        t = float(obs["simulation_time"])
        if self.net is not None and self.needs_launch_decision(obs):
            _, _, ll, _, al = self.forward(self.features(obs))
            if self.launch_rule == "median":
                e = np.exp(ll - ll.max()); self.p_wait *= 1.0 - e[1] / e.sum()
                go = self.p_wait <= 0.5
            else:
                go = int(np.argmax(ll)) == 1
            self.decide_launch(obs, go, al if "attc.w" in self.net else int(np.argmax(al)))
        elif not self.launched() and t >= T_MAX:
            self.decide_launch(obs, True, getattr(self, "att_last", 0))
        self.pre_since += 1
        if self.net is not None and self.p_hat is not None and self._burning(t) and self.needs_decision(obs):
            a = self.choose(obs)
            if self.res_bound and "res.w" in self.net:  # the residual's mean on this decision's features (before decide)
                self.set_res(self.res_mean if self.res_mean is not None else (self.forward(self.features(obs)), self.res_u)[1])
            self.decide(obs, a)
        self.since += 1
        return super(RL2Agent, self).get_action(obs)  # GNC: launches when t >= launch_time

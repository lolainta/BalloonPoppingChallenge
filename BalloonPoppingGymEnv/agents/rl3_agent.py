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
        # 5-output head (RL3_RES2): u[3] -> s_mul = exp(res_sbound tanh u[3]) (flown stretch), u[4] -> n_mul (nav_N), same timing
        self.res_sbound = float(kw.get("res_sbound", (self.net or {}).get("res.sbound", 0.0)))
        self.res_nbound = float(kw.get("res_nbound", (self.net or {}).get("res.nbound", 0.0)))
        # 8-output head (RL3_RES3): u[5:8] -> arrival velocity offset dv = res_vbound tanh u[5:8] (m/s, gnc_agent._arrv)
        self.res_vbound = float(kw.get("res_vbound", (self.net or {}).get("res.vbound", 0.0)))
        self.res_u = None
        # target-conditioned head (a checkpoint with "tc", RL3_TCOND, 2026-10-02): the chosen target's token embedding shifts
        # the stretch logits and the residual mean (see cond); tc_off=1 drops that term (ablation)
        self.tc_off = int(kw.get("tc_off", 0))
        # "tc2" (RL3_TCOND2, 2026-10-03): the same on the policy's second choice (see second); tc2_off=1 drops it (ablation)
        self.tc2_off = int(kw.get("tc2_off", 0))
        # "fast" head (RL3_FAST, 2026-10-05): every fast_k sim steps between two decisions (target / gear held) the policy runs on
        # the current features and the GNC flies u = the decision's u + the fast head's mean (see fast_tick); fast_off=1: no FAST
        # steps (the decision's u held, as without the head; ablation)
        self.fast_k = int(kw.get("fast_k", (self.net or {}).get("fast.k", 0)))
        self.fast_off = int(kw.get("fast_off", 0))
        self.launch_time = np.inf  # set by the policy (or T_MAX)
        self.pre_since = DECIDE
        # how a deterministic run turns the per-decision launch probabilities into a launch:
        #   "argmax": launch when P(launch) > 0.5 at a decision (training samples, so this can wait far longer)
        #   "median": launch once the chance of having launched by now under the policy, 1 - prod(1 - P), reaches 0.5,
        #             i.e. at the median of the policy's own launch time; still decided online from the observations
        self.launch_rule = kw.get("launch_rule", "argmax")
        self.launch_incl = kw.get("launch_incl", 0.0)  # > 0: override the launch attitude (see decide_launch)
        self.launch_head = kw.get("launch_head")  # with launch_incl: this heading (deg, get_initial_attitude) instead of the low-balloon one
        self.launch_at = kw.get("launch_at")  # evaluation probe: launch at the first decision >= this time (s), not the policy's choice
        self.p_wait = 1.0  # prod(1 - P(launch)) over the launch decisions so far
        # redecide=1|2: a post-launch decision that changes the target runs the policy again (same step) on features
        # with the new target flagged as current; 1: take that whole second action, 2: keep the first target, take the
        # second pass's speed gear and residual outputs
        self.redecide = int(kw.get("redecide", 0))
        # near_k=k (evaluation probe, 0 = off): the target is the policy's best among the k live balloons with the
        # shortest predicted reach time (token column 9), as SpaceBerry's "near-two-targets" picks near balloons
        self.near_k = int(kw.get("near_k", 0))
        self.near_col = int(kw.get("near_col", 9))  # 9: the SL model's reach time; 6: GNC's analytic minimum time
        self.near_commit = int(kw.get("near_commit", 0))  # 1: keep the current target until it pops; choose only then
        # ens="a.npz,b.npz": evaluation ensemble, the outputs (target / stretch / launch / attitude logits, value, residual
        # mean, fast mean; tc / tc2 conditioning on each member's own token embeddings, §428) averaged over the main policy
        # and these (same architecture, GNC tunables and SL). Not in make_single_file
        # ens_agg=prob (§505): the categorical heads (target, stretch, launch, discrete attitude) take the mean of the members'
        # probabilities (returned as log-probabilities) instead of the mean of their logits; a member's low logit then no
        # longer vetoes a choice the others make
        self.ens, self.ens_agg = [], str(kw.get("ens_agg", "logit"))
        for path in filter(None, str(kw.get("ens", "") or "").split(",")):
            n = {k: v for k, v in np.load(path).items()}
            assert self.net is not None and n.keys() == self.net.keys() and all(
                n[k].shape == v.shape for k, v in self.net.items()), f"ensemble member {path}: architecture differs"
            assert all(np.array_equal(n[k], v) for k, v in self.net.items() if v.ndim == 0), f"{path}: config differs"
            sib, own = Path(path).parent / Path(path).name.replace("rl3_policy", "rl2_sl"), Path(kw.get("sl_path", ""))
            assert sib.is_file() and own.is_file() and all(np.array_equal(a, b) for a, b in zip(
                np.load(sib).values(), np.load(own).values())), f"{path}: SL {sib} missing or != sl_path {own}"
            self.ens.append(n)

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
                incl, head = self.launch_incl, (None if self.launch_head is None else float(self.launch_head))
            self.incl_head = np.array([incl, self._low_heading(obs, incl) if head is None else head])
            self.q = np.array(get_initial_attitude(*self.incl_head))
        self.pre_since = 0

    def set_res(self, u, d=None):
        """u: a decision's res outputs (held until the next decision, self.u_full); d: a FAST step's delta -> u + d flown, scaled with
        the decision's thrust estimate (a_thrust is refined every step: the held res stays the decision's m/s^2)."""
        u = self.u_full = np.asarray(u, float)
        if d is None:
            self.a_res = self.a_thrust
        else:
            u = u + np.pad(np.asarray(d, float), (0, len(u) - len(d)))  # (a fast head of fewer outputs: u[n:] held)
        self.res = self.res_bound * self.a_res * np.tanh(u[:3])
        self.s_mul = self.n_mul = 1.0  # a 3-output policy (also after a reload from a 5-output one) has no multipliers
        self.dv = np.zeros(3)  # nor a 3- / 5-output one an arrival velocity offset
        if len(u) > 3:
            self.s_mul, self.n_mul = float(np.exp(self.res_sbound * np.tanh(u[3]))), float(np.exp(self.res_nbound * np.tanh(u[4])))
        if len(u) > 5:
            self.dv = self.res_vbound * np.tanh(u[5:8])

    # ------------------------------------------------------------------ numpy network (mirrors dev/rl3_net.py)
    def forward(self, x):
        if not getattr(self, "ens", None):  # (training workers build the agent with __new__: no __init__ attributes)
            return self._forward(x)
        net, outs, us, fus, self.h_toks = self.net, [], [], [], []  # (h_toks: each member's token embeddings, for cond)
        for self.net in [net] + self.ens:
            outs.append(self._forward(x)); us.append(self.res_u); fus.append(getattr(self, "fast_u", None)); self.h_toks.append(self.h_tok)
        self.net = net
        self._cnt("ens")
        self.res_u = None if us[0] is None else np.mean(us, 0)
        self.fast_u = None if fus[0] is None else np.mean(fus, 0)
        if self.ens_agg == "prob":
            self.outs = outs  # (cond combines the stretch head per member: its conditioning term is per member)
            cat = (0, 1, 2) + (() if "attc.w" in net else (4,))
            out = tuple((self.logmean if k in cat else lambda ls: np.mean(ls, 0))([o[k] for o in outs]) for k in range(5))
            int(np.argmax(out[0])) != int(np.argmax(np.mean([o[0] for o in outs], 0))) and self._cnt("ens_dis")
            return out
        return tuple(np.mean([o[k] for o in outs], 0) for k in range(5))

    @staticmethod
    def logmean(ls):
        """ens_agg=prob: log of the mean over members of softmax(logits); masked (-1e9) entries stay -1e9."""
        ps = []
        for l in ls:
            p = np.exp(l - l.max()); ps.append(p / p.sum())
        return np.where(np.mean(ls, 0) > -5e8, np.log(np.maximum(np.mean(ps, 0), 1e-300)), -1e9)

    def _forward(self, x):
        """-> (target logits, stretch logits, launch logits (2,), value, attitude logits (len(ATT),) or attitude mean u (2,)).
        A "res" head's mean u (3,), (5,) or (8,) is left in self.res_u."""
        P = self.net
        g, tok = x[:G_DIM], x[G_DIM:].reshape(N_MAX, F)
        relu = lambda z: np.maximum(z, 0)
        ge = relu(g @ P["g.w"] + P["g.b"])
        h = self.h_tok = RL2Agent._trunk(self, tok, ge)  # (kept for cond)
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
            w, b = P["res.w"], P["res.b"]
            # 3 / 5 / 8 outputs in blocks of 3, 2, 3: bit-equal to the smaller heads they were grown from
            self.res_u = np.concatenate([pg @ w[:, i:j] + b[i:j] for i, j in ((0, 3), (3, 5), (5, 8)) if i < w.shape[1]])
        if "fast.w" in P:  # a FAST step's delta mean: pooled + the current target's (flagged token) embedding (= dev/rl3_net.py)
            self.fast_u = pg @ P["fast.w"] + P["fast.b"]
            cur = np.flatnonzero((tok[:, 8] > 0.5) & mask)
            if cur.size:
                self.fast_u = self.fast_u + (h[cur[0]] @ P["fastc.w"] + P["fastc.b"])
        if "attc.w" in P:  # continuous attitude: the mean of u (2,)
            al = pg @ P["attc.w"] + P["attc.b"]
        else:
            al = pg @ P["att.w"] + P["att.b"] if "att.w" in P else np.eye(len(ATT))[0]  # no head: vertical
        return tl, sl, ll, val, al

    def tc_on(self):
        return "tc.w" in self.net and not getattr(self, "tc_off", 0)  # (training workers skip __init__)

    def tc2_on(self):
        return "tc2.w" in self.net and not getattr(self, "tc2_off", 0)

    def fast_on(self):
        return "fast.w" in self.net and getattr(self, "fast_k", 0) > 0 and not getattr(self, "fast_off", 0)

    def fast_tick(self, obs):
        """A FAST step (RL3_FAST) is due: post-launch, burning, a target held, every fast_k steps since the decision (the decision
        itself, also a pop / target death, comes first: needs_decision). = dev/rl2_env.GuidanceEnv.step's FAST break."""
        return self.cmd is not None and self.since > 0 and self.since % self.fast_k == 0

    @staticmethod
    def second(tl, i):
        """The policy's second choice: argmax of the target logits tl (dead balloons -1e9) over the live balloons other than i;
        None with fewer than 2 alive. (= dev/rl3_net.py; first index on ties, as torch.argmax)"""
        z = np.where(tl > -5e8, tl, -np.inf); z[i] = -np.inf
        return int(np.argmax(z)) if np.isfinite(z).any() else None

    def cond(self, i, sl, tl=None):
        """Post-launch, target i chosen (training: sampled, evaluation: argmax), after forward(): -> the stretch logits
        conditioned on it; the residual mean self.res_u is conditioned in place. (= dev/rl3_net.py, tc on h[i]; tc2 on
        h[second(tl, i)], added after tc; the second choice is left in self.j2)"""
        self.j2 = self.second(tl, i) if self.tc2_on() else None
        k = len(sl) + 5  # (8 residual outputs: the columns of the first 5 on their own, as in forward)
        # ens: the members' mean of their own conditioning terms (cond is additive, so this = the mean of the conditioned outputs)
        nets = list(zip([self.net] + self.ens, self.h_toks)) if getattr(self, "ens", None) else [(self.net, self.h_tok)]
        prob = len(nets) > 1 and self.ens_agg == "prob"
        sls = [o[1].copy() for o in self.outs] if prob else None  # ens_agg=prob: each member's own conditioned stretch logits
        for name, j in (("tc", i if self.tc_on() else None), ("tc2", self.j2)):
            if j is None:
                continue
            cs = [np.concatenate([h[j] @ P[f"{name}.w"][:, :k] + P[f"{name}.b"][:k], h[j] @ P[f"{name}.w"][:, k:] + P[f"{name}.b"][k:]])
                  for P, h in nets]
            c = cs[0] if len(cs) == 1 else np.mean(cs, 0)
            if "res.w" in self.net:
                self.res_u = self.res_u + c[len(sl):]
            if prob:
                for m, cm in zip(sls, cs):
                    m += cm[: len(sl)]
            sl = sl + c[: len(sl)]
        return self.logmean(sls) if prob else sl

    def _value(self, x):
        return self.forward(x)[3]

    def choose(self, obs, tgt=None):
        """tgt: keep this target (redecide=2), the stretch / residual conditioned on it (tc), no search / DP."""
        self.res_mean = None
        for on, h in ((self.tc_on(), "tc"), (self.tc2_on(), "tc2"), (self.fast_on(), "fast")):
            if on and (self.search_k > 0 or self.dp or self.dp_shadow):
                raise NotImplementedError(f"a {h} head conditions on the policy's argmax target: no search_k / DP planner")
        a = self._dp_choose(obs) if self.dp == 1 and tgt is None else None  # DP planner (RL2Agent._dp_choose)
        if a is not None:
            return a
        if self.near_commit and tgt is None and self.cmd is not None and obs["balloon_status"][self.cmd[0], 0] == 1:
            tgt = self.cmd[0]
        x = self.features(obs)
        tl, sl, *_ = self.forward(x)
        if self.near_k and tgt is None:
            tok = x[-N_MAX * F:].reshape(N_MAX, F)
            t1 = np.where(tok[:, 14] > 0.5, tok[:, self.near_col], np.inf)
            far = np.argsort(t1)[self.near_k:]
            tl = tl.copy(); tl[far] = -1e9
        sl = self.cond(int(np.argmax(tl)) if tgt is None else int(tgt), sl, tl)
        self.res_mean = self.res_u  # before _search's look-ahead forwards overwrite it
        s = int(np.argmax(sl))
        if tgt is not None:
            return int(tgt), s
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
            if self.launch_at is not None:
                go = t >= float(self.launch_at) - 1e-6
            self.decide_launch(obs, go, al if "attc.w" in self.net else int(np.argmax(al)))
        elif not self.launched() and t >= T_MAX:
            self.decide_launch(obs, True, getattr(self, "att_last", 0))
        self.pre_since += 1
        if self.net is not None and self.p_hat is not None and self._burning(t) and self.needs_decision(obs):
            a = self.choose(obs)
            if self.redecide and (self.cmd is None or a[0] != self.cmd[0]):
                cmd = self.cmd
                self.cmd = (int(a[0]), STRETCH[int(a[1])])  # the new target flagged as current in the features
                a2 = self.choose(obs, tgt=a[0] if self.redecide == 2 else None)  # 2: conditioned on the kept target
                self.cmd = cmd
                self._cnt("redecide"); a2[0] != a[0] and self._cnt("redecide_tgt"); a2[1] != a[1] and self._cnt("redecide_gear")
                a = a2
            if self.res_bound and "res.w" in self.net:  # the residual's mean on this decision's features (before decide)
                self.set_res(self.res_mean if self.res_mean is not None else (self.forward(self.features(obs)), self.res_u)[1])
            self.decide(obs, a)
        elif self.net is not None and self.p_hat is not None and self.res_bound and self.fast_on() and self._burning(t) \
                and self.fast_tick(obs):  # target / gear held; the decision's u + the fast head's mean on the current features
            self.forward(self.features(obs))
            self.set_res(self.u_full, self.fast_u)
            self._cnt("fast")
        self.since += 1
        return super(RL2Agent, self).get_action(obs)  # GNC: launches when t >= launch_time

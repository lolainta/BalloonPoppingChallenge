"""RL guidance v2: attention policy over ALL balloons + global radar features (+ optional search).

Observation = global vector (G) + one token per balloon slot (N_MAX x F), slot i = balloon i.
  global: rocket v, altitude, burn left, thrust accel (6) | radar: counts of alive balloons by
          azimuth(8) x above/below(2) x range(3) (48) | column stats: mean/std rel pos, frac above, n alive (8)
  token : rel pos, rel vel, analytic reach time t1, altitude, is-current-target,
          t_hat / p_hat (learned reach time / hit prob, analytic until an SL model is present),
          t2 (best reach time from this balloon to the next), neighbours within 30 m / 60 m, alive flag
Action = (balloon slot, stretch level). Weights in rl2_policy.npz (dev/train_ppo2.py), numpy inference here.
Uses only observations and given parameters.
"""

import time
from pathlib import Path

import numpy as np

from BalloonPoppingGymEnv.agents.dp_plan import dp_plan
from BalloonPoppingGymEnv.agents.gnc_agent import GNCAgent, quat_to_R

N_MAX = 100
STRETCH = (0.6, 0.8, 1.0, 1.3)
DECIDE = 50
F_BASE = 15  # the original token columns (the SL's inputs and its decision logs use only these)
F = 18  # + 15..17: the multi-balloon chain (DECISION_LOG §288), zero unless the policy file carries feat.chain
G_DIM = 62
N_OBS = G_DIM + N_MAX * F
T2_GRID = np.arange(0.25, 15.0, 0.25)


GNC_TUNABLES = ("k_att", "max_rate", "k_rate", "nav_N", "t_go_min", "coast", "min_alt", "lag_comp", "bwind", "margin",
                "eob", "eob_T", "eob_tau", "eob_hold",  # flight tunables a checkpoint may carry (eob*: end-of-burn mode, §135)
                "ho_keep", "ho_zone", "eob_pick", "eob_pick_r", "eob_align",  # eval options (gnc_agent), default off
                "att_fix",  # GNSS-aided attitude correction gain (scenario 40, §245), default 0 = off
                "nav_avg", "nav_pad")  # GNSS-velocity dead reckoning + offset mean for the position (§422), default off


def parse_gnc_kw(text):
    """"k_att:3.0+max_rate:2.5" -> {"k_att": 3.0, "max_rate": 2.5} (no "," or "=": safe inside sbatch --export)."""
    out = {}
    for item in filter(None, (text or "").split("+")):
        k, v = item.split(":")
        assert k in GNC_TUNABLES, f"unknown GNC tunable {k!r}"
        out[k] = float(v)
    return out


def reach_time(p, v, bp, bv, a, grid, drift=None):
    """GNC's constant-force reach time, broadcast: p, v (..., 3); bp, bv (..., M, 3) -> (..., M).
    drift (..., M, T, 3): balloon displacement beyond constant velocity at each grid time (GNC bwind)."""
    tg = grid[:, None]
    miss = (bp - p[..., None, :])[..., None, :] + (bv - v[..., None, :])[..., None, :] * tg
    if drift is not None:
        miss = miss + drift
    miss[..., 2] += 4.905 * grid**2
    ok = np.linalg.norm(miss, axis=-1) <= 0.5 * a * grid**2
    return np.where(ok.any(-1), grid[np.argmax(ok, -1)], np.inf)


class RL2Agent(GNCAgent):
    def __init__(self, given_parameters, **kw):
        super().__init__(given_parameters, **kw)
        self.cmd = None
        self.since = 0
        self.n_popped = 0
        self.sl = None  # optional learned reach model: callable(token feats (M,F), stretch) -> (t_hat, p_hat)
        path = Path(kw.get("policy_path", Path(__file__).with_name("rl2_policy.npz")))
        self.net = None
        if kw.get("use_policy", 1) and path.exists():
            self.net = {k: v for k, v in np.load(path).items()}
        self.chain = int(kw.get("chain", 0))  # §288: compute the chain token columns (a policy file's feat.chain also does)
        self.search_k = int(kw.get("search_k", 0))  # 0 = plain policy argmax; k > 0 = one-step search over top-k
        self.gamma_dec = kw.get("gamma_dec", 0.99)  # PPO discount per decision
        sl_path = Path(kw.get("sl_path", Path(__file__).with_name("rl2_sl.npz")))
        if kw.get("use_sl", 1) and sl_path.exists():
            from BalloonPoppingGymEnv.agents.rl2_sl import load_sl
            self.sl = load_sl(sl_path)
        # DP planner (dp_plan.py, dev/plans/dp_planner.md), off by default. dp=1: the planner picks the target after
        # launch; dp=2: it overrides the policy's pick only when its plan is better by dp_delta (dp_delta_sw to leave
        # the current target); dp_shadow=1: the policy acts, the planner only runs. dp_log=1 keeps per-replan records
        self.dp, self.dp_shadow = int(kw.get("dp", 0)), int(kw.get("dp_shadow", 0))
        fl = lambda k, d: tuple(float(x) for x in str(kw.get(k, d)).split("/"))  # noqa: E731  '/'-separated sets
        self.dp_cfg = dict(D=int(kw.get("dp_D", 4)), W=int(kw.get("dp_W", 24)), B0=int(kw.get("dp_B0", 16)),
                           Bk=int(kw.get("dp_Bk", 6)), S0=fl("dp_S", "1.05/1.25/1.5"), S1=fl("dp_S1", "1.05/1.3"),
                           lam=float(kw.get("dp_lam", 0.35)), dmiss=float(kw.get("dp_dmiss", 2.0)),
                           mc=int(kw.get("dp_mc", 40)), td=float(kw.get("dp_td", 0.0)), pc=fl("dp_pc", "0.75/0.9/0.93/20"))
        self.dp_every, self.dp_freeze = int(kw.get("dp_every", 25)), float(kw.get("dp_freeze", 0.5))  # (steps), (s)
        self.dp_switch, self.dp_keep = float(kw.get("dp_switch", 0.2)), float(kw.get("dp_keep", 0.3))
        self.dp_delta, self.dp_delta_sw = float(kw.get("dp_delta", 0.25)), float(kw.get("dp_delta_sw", 0.4))
        self.dp_tarr = str(kw.get("dp_exec", "tarr")) != "stretch"  # fly the planned arrival time (else its stretch)
        self.dp_guard, self.dp_log = float(kw.get("dp_guard", 40.0)), int(kw.get("dp_log", 0))  # (ms) slower: keep cmd
        self.dp_drag = int(kw.get("dp_drag", 0))  # 1: legs include air drag against the history-profile wind
        self.dp_t_arr, self.dp_ovr, self.dp_ms, self.dp_fired, self.dp_rec = np.inf, False, [], 0, []
        # ho_T > 0: one extra decision per target (by index) the first time its GNC reach time t_go drops below ho_T s
        self.ho_T, self.ho_done, self.t_go_last = float(kw.get("ho_T", 0.0)), set(), (None, None)

    # ------------------------------------------------------------------ features
    def needs_decision(self, obs):
        st = obs["balloon_status"][:, 0]
        new = self.cmd is None or int((st == 2).sum()) > self.n_popped or st[self.cmd[0]] != 1
        if self.ho_T and not new and self.cmd[0] not in self.ho_done and self.t_go_last[0] == self.cmd[0] \
                and self.t_go_last[1] < self.ho_T:  # (t_go of the previous step: the GNC runs after the decision)
            self.ho_done.add(self.cmd[0]); self._cnt("ho_T")
            new = True
        if self.dp == 1:  # replan every dp_every steps (not in the last dp_freeze s of a leg), at once when passed
            t = float(obs["simulation_time"])
            return new or self._dp_passed(obs, t) or (self.since >= self.dp_every and self.dp_t_arr - t >= self.dp_freeze)
        return new or self.since >= DECIDE or (self.dp_ovr and self._dp_passed(obs, float(obs["simulation_time"])))

    def features(self, obs):
        p, v = self.p_hat, self.v_hat
        st, bs = obs["balloon_status"][:, 0], obs["balloon_states"]
        t = float(obs["simulation_time"])
        a = self.a_thrust - self.margin
        x = np.zeros(N_OBS, np.float32)
        x[:6] = [*(v / 30.0), (p[2] - self.elev) / 100.0, (self.launched_at + self.burn_time - t) / 30.0, self.a_thrust / 15.0]
        alive = np.flatnonzero(st[:N_MAX] == 1)
        if alive.size == 0:
            return x
        bp, bv = bs[alive, :3], bs[alive, 3:]
        rel, relv = bp - p, bv - v
        # radar
        az = ((np.arctan2(rel[:, 1], rel[:, 0]) + np.pi) / (2 * np.pi) * 8).astype(int) % 8
        up = (rel[:, 2] > 0).astype(int)
        rng = np.digitize(np.linalg.norm(rel, axis=1), [40.0, 100.0])
        radar = np.zeros((8, 2, 3))
        np.add.at(radar, (az, up, rng), 1.0)
        x[6:54] = radar.ravel() / 10.0
        x[54:62] = [*(rel.mean(0) / 100.0), *(rel.std(0) / 100.0), up.mean(), alive.size / 100.0]
        # tokens
        if self.bwind >= 2:  # the reach-time feature also follows the balloon-wind profile
            if self.bwind < 4:  # (bwind >= 4 keeps its history profile, updated in get_action)
                self.wprof = self._wind_profile(obs)
            t1 = reach_time(p, v, bp, bv, a, self.t_grid, self._bdrift(bp[:, None, 2], bv[:, None, 2], self.t_grid))
        else:
            t1 = reach_time(p, v, bp, bv, a, self.t_grid)
        t1c = np.where(np.isfinite(t1), t1, 20.0)
        # lookahead: arrive at each balloon (constant-force arrival velocity, as in GNC), then best next reach time
        tc = np.maximum(t1c, 0.5)[:, None]
        arr_p = bp + bv * tc
        arr_v = v + 2 * (bp - p + (bv - v) * tc) / tc
        others = bp[None] + bv[None] * tc[:, :, None]
        t2 = reach_time(arr_p, arr_v, others, np.broadcast_to(bv[None], others.shape), a, T2_GRID)
        t2[np.arange(alive.size), np.arange(alive.size)] = np.inf
        T2 = t2  # (i, j): analytic reach time from the arrival at i to j (the chain's second-leg candidates)
        t2 = np.minimum(t2.min(1), 20.0)
        d = np.linalg.norm(bp[:, None] - bp[None], axis=-1)
        cur = self.cmd[0] if self.cmd is not None else -1
        tok = np.zeros((alive.size, F), np.float32)
        tok[:, 0:3] = rel / 50.0
        tok[:, 3:6] = relv / 20.0
        tok[:, 6] = t1c / 10.0
        tok[:, 7] = (bp[:, 2] - self.elev) / 100.0
        tok[:, 8] = alive == cur
        tok[:, 9] = t1c / 10.0  # t_hat: analytic until an SL model is loaded
        tok[:, 10] = np.isfinite(t1)  # p_hat
        tok[:, 11] = t2 / 10.0
        tok[:, 12] = ((d < 30).sum(1) - 1) / 5.0
        tok[:, 13] = ((d < 60).sum(1) - 1) / 10.0
        tok[:, 14] = 1.0
        if self.sl is not None:
            tok[:, 9], tok[:, 10] = self.sl(tok, x[:6])
            if getattr(self, "chain", 0) or (self.net is not None and "feat.chain" in self.net):  # (training env: kw chain)
                tok[:, 15:18] = self._chain(x[:6], tok, T2, bp, bv, arr_p, arr_v, tc[:, 0], a)
        x[G_DIM:].reshape(N_MAX, F)[alive] = tok
        return x

    def decide(self, obs, action):
        i, s = int(action[0]), int(action[1])
        st = obs["balloon_status"][:, 0]
        old = self.cmd[0] if self.cmd is not None else None
        if st[i] != 1:  # masked in the policy; guard anyway
            alive = np.flatnonzero(st == 1)
            i = int(alive[0]) if alive.size else -1
        dp_cmd = len(action) > 2 and isinstance(action[2], tuple)  # DP: (i, s, (i, stretch, t_arr)); a training env's
        self.cmd = None if i < 0 else (action[2] if dp_cmd else (i, STRETCH[s]))  # RL3 action [i, s, launch, u..] is not
        self.since = 0
        self.n_popped = int((st == 2).sum())
        if self.ho_keep and old is not None and self.cmd is not None and self.cmd[0] != old and st[old] == 1:
            b = obs["balloon_states"][old]  # leaving a live target in the hand-off zone: it stays a must-hit (gnc ho_keep)
            if self._reach_time(self.p_hat, self.v_hat, b[None, :3], b[None, 3:])[0] <= self.ho_zone + 1e-6:
                self.ho_old = old; self._cnt("ho_hand")

    # ------------------------------------------------------------------ numpy network (mirrors dev/rl2_net.py)
    @staticmethod
    def _ln(x, w, b):
        m = x.mean(-1, keepdims=True)
        s = x.var(-1, keepdims=True)
        return (x - m) / np.sqrt(s + 1e-5) * w + b

    def forward(self, x):
        """-> (target logits (N_MAX,), stretch logits (4,), value)."""
        P = self.net
        g = x[:G_DIM]
        tok = x[G_DIM:].reshape(N_MAX, F)
        mask = tok[:, 14] > 0.5
        relu = lambda z: np.maximum(z, 0)
        ge = relu(g @ P["g.w"] + P["g.b"])
        h = self._trunk(tok, ge)
        tl = (h @ P["tgt.w"] + P["tgt.b"])[:, 0] + P["prior"][0] * (-tok[:, 6])
        tl = np.where(mask, tl, -1e9)
        pooled = (h * mask[:, None]).sum(0) / max(mask.sum(), 1)
        pg = np.concatenate([pooled, ge])
        sl = pg @ P["str.w"] + P["str.b"]
        val = (relu(pg @ P["v0.w"] + P["v0.b"]) @ P["v1.w"] + P["v1.b"])[0]
        return tl, sl, val

    CH_I, CH_J, CH_K = 16, 24, 3  # chain: first balloons expanded, successors per state, second legs expanded further

    def _legs(self, fp, fv, tleft, bpt, bvt, t_an, tref):
        """SL legs from states (fp, fv (k, 3), burn left tleft (k,)) to balloons at bpt (k, 3) moving at bvt, analytic reach
        time t_an, the balloons' own token columns tref (k, F) -> (t_hat s, p_hat)."""
        g = np.concatenate([fv / 30.0, (fp[:, 2:3] - self.elev) / 100.0, tleft[:, None] / 30.0,
                            np.full((len(fp), 1), self.a_thrust / 15.0)], 1)
        tk = tref.copy()
        tk[:, 0:3], tk[:, 3:6] = (bpt - fp) / 50.0, (bvt - fv) / 20.0
        tk[:, 6], tk[:, 7], tk[:, 8] = np.minimum(t_an, 20.0) / 10.0, (bpt[:, 2] - self.elev) / 100.0, 0.0
        th, ph = self.sl(tk, g)
        return th * 10.0, ph

    def _chain(self, g6, tok, T2, bp, bv, arr_p, arr_v, tc, a):
        """DECISION_LOG §288: per balloon, the best 3-balloon chain starting with it inside the burn, every leg the SL's
        (t_hat, p_hat) (no critic, no hand-made hit model) -> columns 15 (expected pops / 3), 16 (its time / 10),
        17 (its second leg's time / 10, 0 = none). The CH_I balloons with the best first leg get the chain over their
        CH_J nearest successors (the CH_K best second legs expanded once more); the others only their first leg."""
        n = len(tok); out = np.zeros((n, 3), np.float32)
        t1, p1 = tok[:, 9] * 10.0, tok[:, 10].astype(float)
        burn = float(g6[3]) * 30.0 + 1.0  # (s; a pop just after burnout still counts)
        ok1 = t1 <= burn
        out[:, 0], out[:, 1] = p1 * ok1 / 3.0, t1 / 10.0
        if n < 2:
            return out
        I = np.argsort(-(p1 - 0.02 * t1) * ok1)[:self.CH_I]
        nj = min(self.CH_J, n - 1)
        J = np.argsort(T2[I], 1)[:, :nj]  # (|I|, nj) nearest successors from each arrival (self excluded: inf on the diagonal)
        ii = np.repeat(I, nj); jj = J.ravel()
        tci = tc[ii]
        t2, p2 = self._legs(arr_p[ii], arr_v[ii], np.maximum(burn - t1[ii], 0.0), bp[jj] + bv[jj] * tci[:, None], bv[jj],
                            T2[ii, jj], tok[jj])
        ok2 = (t1[ii] + t2 <= burn) & np.isfinite(T2[ii, jj])
        v2 = (p2 * ok2).reshape(len(I), nj)
        K = np.argsort(-v2, 1)[:, :min(self.CH_K, nj)]  # (|I|, k) second legs expanded once more
        ia = np.repeat(np.arange(len(I)), K.shape[1]); ka = K.ravel()
        pi, pj = I[ia], J[ia, ka]  # (m,) first / second balloon
        tcij = np.maximum(np.where(np.isfinite(T2[pi, pj]), T2[pi, pj], 20.0), 0.5)
        at = tc[pi] + tcij  # (m,) analytic time at the second arrival
        bpj = bp[pj] + bv[pj] * tc[pi][:, None]
        a2p = bp[pj] + bv[pj] * at[:, None]
        a2v = arr_v[pi] + 2 * (bpj - arr_p[pi] + (bv[pj] - arr_v[pi]) * tcij[:, None]) / tcij[:, None]
        oth = bp[None] + bv[None] * at[:, None, None]  # (m, n, 3)
        T3 = reach_time(a2p, a2v, oth, np.broadcast_to(bv[None], oth.shape), a, T2_GRID)
        T3[np.arange(len(pi)), pi], T3[np.arange(len(pi)), pj] = np.inf, np.inf
        nk = min(self.CH_J, n - 2)
        if nk > 0:
            Kk = np.argsort(T3, 1)[:, :nk]
            mm = np.repeat(np.arange(len(pi)), nk); kk = Kk.ravel()
            tl = (t1[pi] + t2.reshape(len(I), nj)[ia, ka])[mm]
            t3, p3 = self._legs(a2p[mm], a2v[mm], np.maximum(burn - tl, 0.0), oth[mm, kk], bv[kk], T3[mm, kk], tok[kk])
            v3 = (p3 * ((tl + t3 <= burn) & np.isfinite(T3[mm, kk]))).reshape(len(pi), nk)
            b3 = v3.argmax(1); best3, tb3 = v3.max(1), t3.reshape(len(pi), nk)[np.arange(len(pi)), b3]
        else:
            best3, tb3 = np.zeros(len(pi)), np.zeros(len(pi))
        t2m, p2m = t2.reshape(len(I), nj)[ia, ka], v2[ia, ka]
        val = (p2m * (1.0 + best3)).reshape(len(I), -1)  # (|I|, k): expected pops after the first balloon
        kb = val.argmax(1); r = np.arange(len(I))
        out[I, 0] = p1[I] * ok1[I] * (1.0 + val[r, kb]) / 3.0
        out[I, 1] = (t1[I] + t2m.reshape(len(I), -1)[r, kb] + np.where(best3.reshape(len(I), -1)[r, kb] > 0, tb3.reshape(len(I), -1)[r, kb], 0.0)) / 10.0
        out[I, 2] = np.where(val[r, kb] > 0, t2m.reshape(len(I), -1)[r, kb], 0.0) / 10.0
        return out

    def _trunk(self, tok, ge):
        """Token encoder + attention blocks -> (N_MAX, D)."""
        P = self.net
        mask = tok[:, 14] > 0.5
        relu = lambda z: np.maximum(z, 0)
        pre = tok[:, :P["t0.w"].shape[0]] @ P["t0.w"] + P["t0.b"]  # (a 15-column policy ignores the chain columns)
        if "t0x.w" in P:  # §288: the chain columns' own input weights
            pre = pre + tok[:, F_BASE:F] @ P["t0x.w"]
        h = relu(pre) @ P["t1.w"] + P["t1.b"] + ge
        n_layers = int(P["n_layers"])
        heads = int(P["heads"])
        d = h.shape[-1]
        dh = d // heads
        for L in range(n_layers):
            q = f"l{L}."
            z = self._ln(h, P[q + "ln1.w"], P[q + "ln1.b"])
            qkv = z @ P[q + "in.w"] + P[q + "in.b"]
            Q, Kk, V = np.split(qkv, 3, axis=-1)
            Q = Q.reshape(N_MAX, heads, dh).transpose(1, 0, 2)
            Kk = Kk.reshape(N_MAX, heads, dh).transpose(1, 0, 2)
            V = V.reshape(N_MAX, heads, dh).transpose(1, 0, 2)
            att = Q @ Kk.transpose(0, 2, 1) / np.sqrt(dh)
            att = np.where(mask[None, None, :], att, -1e9)
            att = np.exp(att - att.max(-1, keepdims=True))
            att /= att.sum(-1, keepdims=True)
            o = (att @ V).transpose(1, 0, 2).reshape(N_MAX, d)
            h = h + o @ P[q + "out.w"] + P[q + "out.b"]
            z = self._ln(h, P[q + "ln2.w"], P[q + "ln2.b"])
            h = h + relu(z @ P[q + "ff1.w"] + P[q + "ff1.b"]) @ P[q + "ff2.w"] + P[q + "ff2.b"]
        return h

    # ------------------------------------------------------------------ GNC hooks
    def _select_target(self, p, v, obs):
        if self.cmd is None or obs["balloon_status"][self.cmd[0], 0] != 1:
            return None, None
        if len(self.cmd) > 2 and self.cmd[2] is not None:  # DP leg: arrive at t_arr, aimed on the planner's table
            self.stretch, self.aim_prof = 1.0, self.hprof
            return self.cmd[0], max(self.cmd[2] - float(obs["simulation_time"]), self.t_go_min)
        self.aim_prof = None
        tgt, self.stretch = self.cmd[:2]
        b = obs["balloon_states"][tgt]
        t1 = self._reach_time(p, v, b[None, :3], b[None, 3:])[0]
        return tgt, (t1 if np.isfinite(t1) else self.t_grid[-1])

    def choose(self, obs):
        a = self._dp_choose(obs) if self.dp == 1 else None
        if a is not None:
            return a
        x = self.features(obs)
        tl, sl, _ = self.forward(x)
        s = int(np.argmax(sl))
        a = (int(np.argmax(tl)) if self.search_k <= 0 else self._search(obs, x, tl, s)), s
        return self._dp_choose(obs, tl, a) if self.dp or self.dp_shadow else a

    # ------------------------------------------------------------------ DP planner (dp_plan.py)
    def _dp_passed(self, obs, t):
        """The committed target's closest approach is behind us, after its planned arrival time."""
        b = obs["balloon_states"][self.cmd[0]]
        return t >= self.dp_t_arr and (b[:3] - self.p_hat) @ (b[3:] - self.v_hat) > 0

    def _dp_choose(self, obs, tl=None, a=None):
        """dp=1: plan and pick (None: no plan, the policy decides); dp=2 / dp_shadow: keep or override the policy's
        pick a. -> action (i, s, cmd) with cmd = (i, 1, t_arr) (dp_exec=tarr) or (i, stretch, None)."""
        st, t = obs["balloon_status"][:, 0], float(obs["simulation_time"])
        passed = self.cmd is not None and st[self.cmd[0]] == 1 and self._dp_passed(obs, t)
        cur = self.cmd[0] if self.cmd is not None and st[self.cmd[0]] == 1 and not passed else -1
        self.dp_ovr &= cur >= 0
        t0 = time.perf_counter()
        alive, B = self._btable(obs)
        res, Jm = None, {}
        if alive.size:
            ai = {int(g): j for j, g in enumerate(alive)}
            force = [] if tl is None else [ai[int(g)] for g in np.argsort(-tl)[:3 if self.dp == 2 else 1] if int(g) in ai]
            kc = int(round((self.dp_t_arr - t) / 0.1)) if cur >= 0 and np.isfinite(self.dp_t_arr) else 0
            cfg = dict(self.dp_cfg, B0=8) if self.dp == 2 else dict(self.dp_cfg)  # dp=2 roots: policy top-3 + physics top-8
            if self.dp_drag and self.hprof is not None:
                rho = 1.225 * np.clip(1.0 - 2.2558e-5 * self.p_hat[2], 0.5, 1.0) ** 4.2559
                cfg["drag"] = (self.k_drag * rho, self.hprof[0], self.hprof[1])
            res = dp_plan(B, 0.1, self.p_hat, self.v_hat, quat_to_R(self.q)[:, 2], self.launched_at + self.burn_time - t,
                          self.a_thrust - self.margin, 0.8 * self.max_rate, self.elev, cfg, ai.get(cur, -1), kc, force)
            if res is not None and np.isfinite(res["val"]).any():
                for j in np.argsort(res["val"]):  # best plan value through each first balloon (with a full plan)
                    if np.isfinite(res["val"][j]):
                        Jm[int(alive[res["m"][j]])] = res["val"][j]
            else:
                res = None
        ms = (time.perf_counter() - t0) * 1e3
        slow = ms > self.dp_guard > 0
        self.dp_ms.append(ms)
        self.dp_fired += slow
        g_pi = a[0] if a is not None else -1
        pick = None  # index of the flown root leg
        if res is not None and (not slow or (self.dp == 1 and cur < 0)):  # guard: keep the command / the policy's pick
            val, m = res["val"], res["m"]
            b = int(np.argmax(val))
            gb, Jb = int(alive[m[b]]), val[b]
            if self.dp == 1:
                ic = [j for j in range(len(m)) if alive[m[j]] == cur and np.isfinite(val[j])]
                if ic:  # switch only for dp_switch more; same target: keep the committed arrival if close and feasible
                    jc = max(ic, key=lambda j: val[j])
                    b = b if Jb >= val[jc] + self.dp_switch else jc
                    com = [j for j in ic if res["com"][j]]
                    if alive[m[b]] == cur and com and abs(res["k"][b] - kc) * 0.1 < self.dp_keep:
                        b = com[0]
                pick = b
            elif self.dp == 2:
                if self.dp_ovr:  # an override is held until pop / pass unless the planner clears dp_delta_sw
                    pick = b if gb != cur and Jb - Jm.get(cur, -np.inf) > self.dp_delta_sw else None
                elif gb != g_pi and g_pi in Jm and Jb - Jm[g_pi] > max(self.dp_delta, self.dp_delta_sw * (0 <= cur != gb)):
                    pick = b
        if pick is not None:
            g, k, s = int(alive[res["m"][pick]]), int(res["k"][pick]), float(res["s"][pick])
            self.dp_t_arr, self.dp_ovr = t + k * 0.1, self.dp == 2
            out = (g, 0, (g, 1.0, self.dp_t_arr) if self.dp_tarr else (g, s, None))
        elif cur >= 0 and (self.dp_ovr or (self.dp == 1 and self.cmd is not None)):
            out = (cur, 0, self.cmd)  # hold the override / keep the command (guard)
        else:
            self.dp_t_arr, self.dp_ovr, out = np.inf, False, a
        if self.dp_log:
            self.dp_rec.append(dict(t=t, ms=ms, slow=slow, passed=passed, n_pop=int((st == 2).sum()), pi=g_pi,
                                    tgt=None if out is None else out[0], t_arr=self.dp_t_arr, ovr=self.dp_ovr,
                                    J=None if res is None else float(np.max(res["val"])), Jpi=Jm.get(g_pi), cur=cur, Jcur=Jm.get(cur),
                                    V=None if pick is None else res["V"][pick].copy(), s=None if pick is None else float(res["s"][pick]),
                                    t_end=None if res is None or not len(res["seq"]) else t + (res["seq"][-1][1]) * 0.1,
                                    Ep=None if res is None else res["Ep"], nodes=0 if res is None else res["nodes"]))
        return out

    def _value(self, x):
        return self.forward(x)[2]

    def _search(self, obs, x, tl, s):
        """One-step lookahead: for the policy's top-k targets, predict the hit (prob, time, arrival state),
        rebuild the observation there and score  p_hit * (1 + gamma^n * V(next))."""
        tok = x[-N_MAX * F :].reshape(N_MAX, F)  # tokens are last (RL3 has a longer global part)
        top = [int(i) for i in np.argsort(-tl)[: self.search_k] if tok[i, 14] > 0.5]
        if len(top) <= 1:
            return top[0] if top else int(np.argmax(tl))
        p0, v0, cmd0 = self.p_hat.copy(), self.v_hat.copy(), self.cmd
        t = float(obs["simulation_time"])
        bs, st = obs["balloon_states"], obs["balloon_status"]
        best, best_score = top[0], -np.inf
        for i in top:
            tau = max(float(tok[i, 9]) * 10.0 * (STRETCH[s] if self.sl is None else 1.0), 0.3)
            p_hit = float(tok[i, 10])
            nb = bs.copy()
            nb[:, :3] = bs[:, :3] + bs[:, 3:] * tau
            ns = st.copy()
            ns[i, 0] = 2
            rel = bs[i, :3] - p0 + (bs[i, 3:] - v0) * tau
            self.p_hat, self.v_hat, self.cmd = nb[i, :3].copy(), v0 + 2 * rel / tau, None
            burning = self.launched_at + self.burn_time - (t + tau) > 0
            val = 0.0
            if burning:
                val = self._value(self.features({"simulation_time": t + tau, "balloon_states": nb, "balloon_status": ns}))
            score = p_hit * (1.0 + self.gamma_dec ** (tau / 0.5) * val)
            if score > best_score:
                best, best_score = i, score
        self.p_hat, self.v_hat, self.cmd = p0, v0, cmd0
        return best

    def get_action(self, obs):
        t = float(obs["simulation_time"])
        if self.net is not None and self.p_hat is not None and self._burning(t) and self.needs_decision(obs):
            self.decide(obs, self.choose(obs))
        self.since += 1
        return super().get_action(obs)

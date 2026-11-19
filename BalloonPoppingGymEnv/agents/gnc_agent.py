"""Baseline GNC agent: greedy target selection + ZEM guidance + cascaded attitude control.

Uses only `observation` and `given_parameters` (no seed, no simulator internals).

Navigation : GNSS position/velocity as-is, attitude by integrating the gyro from the
             launch attitude, thrust acceleration estimated from the accelerometer.
Guidance   : pick the released balloon with the lowest (intercept time + turn penalty),
             steer with zero-effort-miss guidance plus a closing-speed term.
Control    : desired specific force -> desired nose direction -> body-rate command -> TVC.
"""

import numpy as np

from BalloonPoppingGymEnv.agents.base_agent import BaseAgent
from BalloonPoppingGymEnv.envs.balloon_world import get_initial_attitude

G = np.array([0.0, 0.0, -9.81])


def quat_to_R(q):
    """Body -> launch frame rotation, q = (e0, e1, e2, e3) scalar first."""
    e0, e1, e2, e3 = q
    return np.array(
        [
            [1 - 2 * (e2 * e2 + e3 * e3), 2 * (e1 * e2 - e0 * e3), 2 * (e1 * e3 + e0 * e2)],
            [2 * (e1 * e2 + e0 * e3), 1 - 2 * (e1 * e1 + e3 * e3), 2 * (e2 * e3 - e0 * e1)],
            [2 * (e1 * e3 - e0 * e2), 2 * (e2 * e3 + e0 * e1), 1 - 2 * (e1 * e1 + e2 * e2)],
        ]
    )


def quat_step(q, w, dt):
    """q <- q ⊗ exp(w dt / 2), w in body frame."""
    v = w * dt
    a = np.linalg.norm(v)
    if a < 1e-12:
        return q
    dw, (dx, dy, dz) = np.cos(a / 2), np.sin(a / 2) * v / a
    e0, e1, e2, e3 = q
    q = np.array(
        [
            e0 * dw - e1 * dx - e2 * dy - e3 * dz,
            e0 * dx + e1 * dw + e2 * dz - e3 * dy,
            e0 * dy - e1 * dz + e2 * dw + e3 * dx,
            e0 * dz + e1 * dy - e2 * dx + e3 * dw,
        ]
    )
    return q / np.linalg.norm(q)


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v * 0.0


class GNCAgent(BaseAgent):
    def __init__(self, given_parameters, **kw):
        super().__init__(given_parameters)
        p = self.given_parameters
        rk, ctrl = p["rocket"], p["rocket"]["control"]
        self.dt = p["simulation"]["time_step"]
        self.elev = p["environment"]["elevation"]
        self.burn_time = rk["motor"]["burn_time"]
        self.max_gimbal = ctrl["max_gimbal_angle"]
        self.max_roll = ctrl["max_roll_torque"]
        self.thr_lo, self.thr_hi = ctrl["throttle_range"]
        self.thr_rate = ctrl["throttle_rate_limit"]

        # tunables (all overridable from cfg.yaml agent_kwargs)
        self.launch_time = kw.get("launch_time", 20.0)
        self.incl_head = np.array([kw.get("inclination", 90.0), kw.get("heading", 0.0)])
        self.margin = kw.get("margin", 0.5)  # (m/s^2) thrust held back when planning
        self.stretch = kw.get("stretch", 1.3)  # fly t_go = stretch * min time (less speed build-up)
        self.switch_margin = kw.get("switch_margin", 0.3)  # (s) hysteresis
        self.switch_ratio = kw.get("switch_ratio", 1.15)  # keep target unless another is this much faster
        self.excess_dir = kw.get("excess_dir", 0)  # see _spend_excess
        self.excess_frac = kw.get("excess_frac", 1.0)
        self.nav_N = kw.get("nav_N", 2.0)  # ZEM guidance gain
        self.t_go_min = kw.get("t_go_min", 0.3)  # (s) guidance singularity guard
        self.t_grid = np.arange(0.1, kw.get("horizon", 15.0), 0.1)
        self.n_cand = int(kw.get("n_cand", 8))  # candidates expanded by the lookahead
        self.gamma = kw.get("gamma", 0.5)  # weight of the second leg
        self.min_alt = kw.get("min_alt", 30.0)  # (m AGL) ground avoidance floor
        self.k_att = kw.get("k_att", 2.0)  # (rad/s per rad) attitude -> rate
        self.max_rate = kw.get("max_rate", 1.5)  # (rad/s)
        self.k_rate = kw.get("k_rate", 10.0)  # (deg per rad/s) rate -> gimbal
        self.k_pos = kw.get("k_pos", 0.02)  # GNSS position blend per step
        self.k_vel = kw.get("k_vel", 0.05)  # GNSS velocity blend per step
        # att_fix > 0 (DECISION_LOG §245, scenario 40's 4x gyro noise / 10x drift): GNSS-aided tilt correction of the
        # gyro-integrated attitude, this gain per step; 0 = gyro only (as before)
        self.att_fix = float(kw.get("att_fix", 0.0))
        # nav_avg > 0 (§422): the GNSS velocity (0.1 m/s) is far better than its position (3 m) -> position = dead reckoning
        # on the GNSS velocity + the running mean of the GNSS position offsets, gain max(1/n, nav_avg); nav_pad = weight (in
        # GNSS samples) of the known pad position (0, 0, elev) at the first fix, 0 = start at the fix
        self.nav_avg, self.nav_pad = float(kw.get("nav_avg", 0.0)), float(kw.get("nav_pad", 0.0))
        self.k_roll = kw.get("k_roll", 20.0)  # (N m per rad/s)
        self.coast = kw.get("coast", 0)  # 1: plan hits past burnout (thrust only for the burn left, then ballistic)
        # coast >= 2: burn-tail coast planner (see _coast_plan)
        self.plan_T = kw.get("plan_T", 4.0)  # (s) burn left when the planner may take over
        self.align_T = kw.get("align_T", 1.0)  # (s) last burn seconds flown along the velocity: no angle of attack
        self.plan_min = kw.get("plan_min", 0.3)  # expected coast hits needed to take over from the target
        self.plan_every = int(kw.get("plan_every", 20))  # (steps) replanning period
        # eob=1: end-of-burn mode (see _eob): in the last eob_T s of the burn, a target that cannot be hit before burnout
        # (t_go * stretch > burn left) is aimed at a hit <= eob_tau s after it; the thrust is held for the last eob_hold s.
        # eob=2: "cannot be hit before burnout" = no constant force within the thrust hits it by burnout; once on, the
        # mode stays on for that target
        self.eob = int(kw.get("eob", 0))
        self.eob_T = kw.get("eob_T", 2.0)  # (s) burn left when the mode may engage
        self.eob_tau = kw.get("eob_tau", 2.0)  # (s) latest hit time after burnout
        self.eob_hold = kw.get("eob_hold", 0.25)  # (s) last burn seconds with the thrust held
        self.eob_f = None  # the mode's last force (held at the end), None when off
        # eob_pick=1: among the feasible tau prefer those whose ballistic post-hit path passes within eob_pick_r m of
        # another live balloon in the 1.5 s after the hit (min force among them; none -> min force as before)
        self.eob_pick, self.eob_pick_r = int(kw.get("eob_pick", 0)), kw.get("eob_pick_r", 2.0)
        self.eob_align = kw.get("eob_align", 0.0)  # (deg) > 0: hold phase force along the velocity when within this angle
        # ho_keep=r_ok (m, §136 F-group (c)): when a decision leaves a live target with GNC reach time <= ho_zone s, it
        # stays a must-hit: a = (1-l) a_old + l a_new, l in 0..1 with the old predicted to pass within r_ok, min
        # t_hit(old) + t_reach(new) (dev/gnc2_gate/gate.py Fc); none feasible -> a_new, the old is dropped
        self.ho_keep, self.ho_zone, self.ho_old = kw.get("ho_keep", 0.0), kw.get("ho_zone", 1.5), None
        self.opt_n = {}  # trigger counts of the eval options above (and rl3 redecide / ho_T / ens)
        # actuator lag (scenario 3 / 4): each actuator is y = a u + (1 - a) y_prev with a = 1 / (1 + tau / dt), then
        # its rate limit and range; tau and the limits are given parameters. lag_comp=1 tracks y and commands the u
        # (inside the action range) whose output is the wanted value -> the lag is cancelled wherever the range allows
        self.lag_comp = int(kw.get("lag_comp", 0))
        alpha = lambda tau: 1.0 / (1.0 + tau / self.dt) if tau else 1.0  # noqa: E731
        self.act_a = np.array([alpha(ctrl.get("gimbal_time_constant"))] * 2
                              + [alpha(ctrl.get("roll_torque_time_constant")), alpha(ctrl.get("throttle_time_constant"))])
        rate = [ctrl.get("gimbal_rate_limit")] * 2 + [ctrl.get("torque_rate_limit"), self.thr_rate]
        self.act_rate = np.array([np.inf if r is None else r * self.dt for r in rate])
        self.act_lo = np.array([-self.max_gimbal, -self.max_gimbal, -self.max_roll, self.thr_lo])
        self.act_hi = np.array([self.max_gimbal, self.max_gimbal, self.max_roll, self.thr_hi])
        self.act_out = np.array([0.0, 0.0, 0.0, 1.0])  # modelled outputs: tvc x, tvc y, roll torque, throttle
        # bwind=1: balloons drift with the air, so their horizontal velocities give the wind at their altitudes; a
        # balloon's predicted path follows that profile as it rises (scenario 4 gust changes every 20 m of altitude)
        self.bwind = int(kw.get("bwind", 0))
        self.bw_h = kw.get("bw_h", 5.0)  # (m) altitude kernel width of the balloon-wind profile
        self.wz = self.elev + np.arange(0.0, 600.0, 2.0)  # (m ASL) profile grid
        self.wprof = None  # (altitude grid (Z,), wind (Z, 2), its altitude integral (Z, 2))
        # bwind >= 4: the profile from ALL past balloon samples instead of the current snapshot (here the wind depends on
        # altitude only and the balloons are its tracers): local-linear running moments on a 0.5 m grid, every 0.1 s;
        # held constant above the highest balloon seen (no assumption about the gust structure there)
        self.hz = self.elev + np.arange(0.0, 1200.0, 0.5)
        self.hmom, self.hrel, self.hstep = np.zeros((5, len(self.hz), 2)), None, 0
        # DP planner (rl2_agent dp > 0 / dp_shadow): its balloon table uses the history profile hprof, collected here
        # even when bwind < 4 (bwind >= 4 shares it as wprof); aim_prof: the profile a planner leg is aimed with
        self.dp_hist = int(kw.get("dp", 0)) > 0 or int(kw.get("dp_shadow", 0)) > 0
        self.hprof = self.aim_prof = None

        # initial thrust acceleration guess from given parameters, refined in flight
        grain = rk["motor"]
        m_grain = (
            np.pi
            * (grain["grain_outer_radius"] ** 2 - grain["grain_initial_inner_radius"] ** 2)
            * grain["grain_initial_height"]
            * grain["grain_density"]
            * grain["grain_number"]
        )
        m0 = (
            rk["rocket_body"]["mass"]
            + grain["dry_mass"]
            + rk["tank"]["initial_liquid_mass"]
            + rk["tank"]["initial_gas_mass"]
            + m_grain
        )
        self.a_thrust = grain["thrust_source"] / m0
        tank, body = rk["tank"], rk["rocket_body"]
        m_end = body["mass"] + grain["dry_mass"] + tank["initial_gas_mass"] + max(
            tank["initial_liquid_mass"] - tank["liquid_mass_flow_rate_out"] * self.burn_time, 0.0)
        self.k_drag = 0.5 * body["power_off_drag"] * np.pi * body["radius"] ** 2 / m_end  # drag accel = k rho |v|^2
        el = np.radians([-30.0, 0.0, 30.0, 60.0])[:, None]; az = np.radians(np.arange(0, 360, 45))[None]
        dirs = np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el) + 0 * az], -1).reshape(-1, 3)
        self.plan_dirs = np.vstack([dirs, [0.0, 0.0, 1.0]])  # candidate thrust directions (world)
        self.plan, self.plan_on = None, False
        # learned guidance residual (rl3_agent res head): added to the guidance force while burning with a target
        self.res = np.zeros(3)
        # and (5-output head) multipliers of the flown stretch (t_go = t_go * stretch * s_mul) and of nav_N; 1 = off
        self.s_mul = self.n_mul = 1.0
        # and (8-output head) an offset dv (m/s) of the arrival velocity, see _arrv; |its acceleration| <= vcap * a_thrust;
        # dv_p / dv_v: the deviation of position / velocity it has caused so far (this target), dv_tgt: that target
        self.dv, self.dv_p, self.dv_v, self.dv_tgt = np.zeros(3), np.zeros(3), np.zeros(3), None
        self.vcap = kw.get("vcap", 0.5)
        # v2t > 0 (DECISION_LOG §199, SpaceBerry-style "near two targets"): arrive at the target moving toward the next one
        # (the best second leg of the lookahead): dv = v2t * (|v_arr| u_next - v_arr), |dv| <= v2t_max (pure GNCAgent only)
        self.v2t, self.v2t_max, self.next2 = float(kw.get("v2t", 0.0)), float(kw.get("v2t_max", 10.0)), None

        self.q =np.array(get_initial_attitude(*self.incl_head))
        self.launched_at = None
        self.p_hat = self.v_hat = None
        self.throttle = 1.0  # our model of the actual (rate limited) throttle
        self.target = None
        self.debug = {}

    # ------------------------------------------------------------------ helpers
    def _navigate(self, acc, p_gnss, v_gnss):
        """Complementary filter: INS propagation (accelerometer + attitude), GNSS correction."""
        if self.p_hat is None:
            self.p_hat, self.v_hat = p_gnss.copy(), v_gnss.copy()
            if self.nav_avg > 0:
                self.nav_n, self.v_prev = self.nav_pad + 1.0, v_gnss.copy()
                pad = np.array([0.0, 0.0, self.elev])
                self.p_hat = pad + (p_gnss - pad) / self.nav_n  # (nav_pad 0: the fix itself)
            return self.p_hat, self.v_hat
        if self.att_fix > 0:
            # a tilt delta of the estimate makes the velocity innovation settle at ~ (delta x f) dt / k_vel (f: the predicted
            # world specific force) -> delta ~ k_vel / dt (f x innovation) / |f|^2; while thrusting (|f| > 3 m/s^2) rotate the
            # estimate by att_fix * delta (the roll about f is unobservable this way and left alone)
            Rq = quat_to_R(self.q); f = Rq @ acc; ff = f @ f
            if ff > 9.0:
                d = self.k_vel / self.dt * np.cross(f, v_gnss - self.v_hat) / ff
                self.q = quat_step(self.q, Rq.T @ (self.att_fix * d) / self.dt, self.dt)  # world rotation -> body rate
        a = quat_to_R(self.q) @ acc + G
        if self.nav_avg > 0:
            self.nav_n += 1.0
            self.p_hat = self.p_hat + 0.5 * (self.v_prev + v_gnss) * self.dt
            self.p_hat += max(1.0 / self.nav_n, self.nav_avg) * (p_gnss - self.p_hat)
            self.v_prev = v_gnss.copy()
        else:
            self.p_hat = self.p_hat + self.v_hat * self.dt + 0.5 * a * self.dt**2
            self.p_hat += self.k_pos * (p_gnss - self.p_hat)
        self.v_hat = self.v_hat + a * self.dt
        self.v_hat += self.k_vel * (v_gnss - self.v_hat)
        return self.p_hat, self.v_hat

    def _burning(self, t):
        return self.launched_at is not None and t < self.launched_at + self.burn_time - self.dt

    def _wind_profile(self, obs):
        """Horizontal wind vs altitude from the airborne balloons: Gaussian kernel (bw_h) over altitude, held constant
        beyond the balloons -> (W, integral of W dz) on self.wz, or None."""
        st, bs = obs["balloon_status"][:, 0], obs["balloon_states"]
        fly = (st == 1) & (bs[:, 2] > self.elev + 1.0)
        if fly.sum() < 3:
            return None
        z, w = bs[fly, 2], bs[fly, 3:5]
        K = np.exp(-0.5 * ((self.wz[:, None] - z[None]) / self.bw_h) ** 2)
        ks = K.sum(1)
        ok = ks > 0.05
        W = (K[ok] @ w) / ks[ok, None]
        W = np.stack([np.interp(self.wz, self.wz[ok], W[:, k]) for k in (0, 1)], 1)
        Wc = np.vstack([np.zeros((1, 2)), np.cumsum(0.5 * (W[1:] + W[:-1]) * np.diff(self.wz)[:, None], 0)])
        return self.wz, W, Wc

    def _hist_add(self, obs):
        """bwind >= 4: add this observation's settled balloons (released >= 0.3 s ago) to the running moments, rebuild."""
        st, bs, t = obs["balloon_status"][:, 0], obs["balloon_states"], float(obs["simulation_time"])
        if self.hrel is None:
            self.hrel = np.full(len(st), np.nan)
        self.hrel[np.isnan(self.hrel) & (st == 1)] = t
        ok = (st == 1) & (t - self.hrel >= 0.3) & (bs[:, 2] > self.elev + 1.0) & np.isfinite(bs[:, 2:5]).all(1)  # one NaN would poison the sums
        if not ok.any():
            return
        z, v = bs[ok, 2].astype(float), bs[ok, 3:5].astype(float)
        dz = self.hz[1] - self.hz[0]
        i = np.clip((z - self.hz[0]) / dz, 0, len(self.hz) - 1.001).astype(int)
        for j in (i, i + 1):  # hat weights to the two neighbouring nodes; moments about each node (local line fit)
            d = z - self.hz[j]; w = 1.0 - np.abs(d) / dz
            for c, q in enumerate((w, w * d, w * d * d)):
                self.hmom[c] += np.bincount(j, q, len(self.hz))[:, None]
            for c, q in ((3, w), (4, w * d)):
                self.hmom[c] += np.stack([np.bincount(j, q * v[:, k], len(self.hz)) for k in (0, 1)], 1)
        s0, s1, s2, t0, t1 = self.hmom
        have = s0[:, 0] > 1e-6
        if have.sum() < 2:
            return
        det = s0 * s2 - s1 * s1
        lin = (det > 1e-4 * s0 * s0 * dz ** 2)[:, 0]  # enough altitude spread in the bin for a local line
        slope = (s0 * t1 - s1 * t0) / np.where(det > 0, det, 1.0)
        near = np.maximum.accumulate(np.where(lin, np.arange(len(lin)), 0))  # else the slope of the nearest line below
        val = (t0 - slope[near] * s1) / np.maximum(s0, 1e-12)
        W = np.stack([np.interp(self.hz, self.hz[have], val[have, k]) for k in (0, 1)], 1)  # constant beyond the data
        Wc = np.vstack([np.zeros((1, 2)), np.cumsum(0.5 * (W[1:] + W[:-1]) * dz, 0)])
        self.hprof = (self.hz, W, Wc)
        if self.bwind >= 4:
            self.wprof = self.hprof

    def _bdrift(self, z0, vz, t, wp=None):
        """Extra displacement of balloons at z0 rising at vz over t beyond constant velocity (broadcast) -> (..., 3).
        wp: wind profile to use (default wprof)."""
        wp = self.wprof if wp is None else wp
        if wp is None:
            return np.zeros(np.broadcast(z0, vz, t).shape + (3,))
        wz, W, Wc = wp
        z0, vz, t = np.broadcast_arrays(z0, vz, t)
        z1 = z0 + vz * t
        rising = np.abs(vz) > 0.2
        out = np.zeros(z0.shape + (3,))
        for k in (0, 1):
            dW = np.interp(z1, wz, Wc[:, k]) - np.interp(z0, wz, Wc[:, k])
            out[..., k] = np.where(rising, dW / np.where(rising, vz, 1.0) - np.interp(z0, wz, W[:, k]) * t, 0.0)
        return out

    def _btable(self, obs, H=15.0, dt=0.1, min_alt=3.0):
        """DP planner's balloon table: the airborne balloons (alive, > min_alt AGL) at t = 0, dt, ..., H, drifting on the
        history profile -> (their indices, positions (K + 1, M, 3))."""
        st, bs = obs["balloon_status"][:, 0], obs["balloon_states"]
        alive = np.flatnonzero((st == 1) & (bs[:, 2] > self.elev + min_alt) & np.isfinite(bs).all(1))
        t = np.arange(int(round(H / dt)) + 1) * dt
        bp, bv = bs[alive, :3].astype(float), bs[alive, 3:].astype(float)
        return alive, bp[None] + bv[None] * t[:, None, None] + self._bdrift(bp[None, :, 2], bv[None, :, 2], t[:, None], self.hprof)

    def _reach_time(self, p, v, bpos, bvel):
        """Min time to hit each balloon with constant specific force |f| <= a_thrust - margin.

        Hit at t  <=>  |r + (vb - v) t - G t^2 / 2| <= a t^2 / 2   (drag ignored).
        Broadcasts: p, v (..., 3); bpos, bvel (..., N, 3) -> (..., N). inf if unreachable.
        """
        tg = self.t_grid[:, None]
        r = (bpos - p[..., None, :])[..., None, :]
        vr = (bvel - v[..., None, :])[..., None, :]
        miss = r + vr * tg - 0.5 * G * tg**2
        if self.bwind:
            miss = miss + self._bdrift(bpos[..., None, 2], bvel[..., None, 2], self.t_grid)
        ok = np.linalg.norm(miss, axis=-1) <= (self.a_thrust - self.margin) * self._burn_disp(self.t_grid)
        return np.where(ok.any(axis=-1), self.t_grid[np.argmax(ok, axis=-1)], np.inf)

    def _select_target(self, p, v, obs):
        """Two-step lookahead: cost = t(target) + gamma * min t(target -> next)."""
        status = obs["balloon_status"][:, 0]
        states = obs["balloon_states"]
        alive = np.flatnonzero(status == 1)
        if alive.size == 0:
            return None, None
        bp, bv = states[alive, :3], states[alive, 3:]
        t1 = self._reach_time(p, v, bp, bv)
        cand = np.argsort(t1)[: self.n_cand]
        cand = cand[np.isfinite(t1[cand])]
        if self.target is not None and status[self.target] == 1:
            cur = np.flatnonzero(alive == self.target)[0]
            if np.isfinite(t1[cur]) and cur not in cand:
                cand = np.append(cand, cur)
        if cand.size == 0:
            return None, None
        tc = t1[cand][:, None]
        arr_p = bp[cand] + bv[cand] * tc
        arr_v = v + 2 * (bp[cand] - p + (bv[cand] - v) * tc) / tc  # v + a t
        t2 = self._reach_time(arr_p, arr_v, bp[None] + bv[None] * tc[:, :, None], bv[None])
        t2[np.arange(cand.size), cand] = np.inf  # not itself
        j2 = np.argmin(t2, axis=1)  # (v2t) each candidate's best next balloon
        t2 = np.minimum(t2.min(axis=1), self.t_grid[-1])
        cost = t1[cand] + self.gamma * t2
        i = np.argmin(cost)
        if self.target is not None and status[self.target] == 1:
            j = np.flatnonzero(alive[cand] == self.target)
            if j.size and cost[j[0]] <= self.switch_ratio * cost[i] + self.switch_margin:
                i = j[0]
        k = j2[i]  # (v2t) arrival point / natural arrival velocity at the chosen one, the next one's position then
        self.next2 = (arr_p[i], arr_v[i], bp[k] + bv[k] * tc[i, 0]) if np.isfinite(t2[i]) and t2[i] < self.t_grid[-1] else None
        return alive[cand[i]], t1[cand[i]]

    def _burn_disp(self, t):
        """Displacement per unit specific force held for t (t^2/2), or only for the burn left, then coasting."""
        if not self.coast:
            return 0.5 * t**2
        tb = self.burn_left
        return np.where(t <= tb, 0.5 * t**2, tb * (t - 0.5 * tb))

    def _wind(self, obs):
        """Wind estimate: horizontal drift of the airborne balloons (they move with the air)."""
        st, bs = obs["balloon_status"][:, 0], obs["balloon_states"]
        fly = (st == 1) & (bs[:, 2] > self.elev + 2.0)
        return np.r_[np.median(bs[fly, 3:5], 0), 0.0] if fly.any() else np.zeros(3)

    def _fly(self, p, v, f, t_burn, w, T, dt=0.1, f0=None):
        """Point-mass paths of K candidates: specific force f (K, 3) until t_burn (its last align_T seconds along the
        velocity), then coasting; gravity + drag against wind w. The thrust starts at f0 (current attitude x throttle)
        and turns / ramps at the attitude-rate and throttle-rate limits. -> positions (n + 1, K, 3) at t = 0, dt, ..."""
        n = int(round(T / dt)); P = np.empty((n + 1,) + p.shape); P[0] = p
        fm = np.linalg.norm(f, axis=-1)
        cd = np.broadcast_to(unit(f0) if f0 is not None else np.array([0.0, 0.0, 1.0]), p.shape).copy()  # thrust direction
        cm = np.full(len(p), np.linalg.norm(f0) if f0 is not None else self.a_thrust)  # thrust magnitude
        w_max, m_rate = 0.8 * self.max_rate * dt, self.thr_rate * self.a_thrust * dt
        for i in range(n):
            ti = i * dt
            vr = v - w
            rho = 1.225 * np.clip(1.0 - 2.2558e-5 * p[..., 2:3], 0.5, 1.0) ** 4.2559
            a = G - self.k_drag * rho * np.linalg.norm(vr, axis=-1, keepdims=True) * vr
            if ti < t_burn:
                td = v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-6) if ti >= t_burn - self.align_T \
                    else f / np.maximum(fm[:, None], 1e-9)
                th = np.arccos(np.clip((cd * td).sum(-1), -1.0, 1.0))
                al = np.minimum(th, w_max); sn = np.maximum(np.sin(th), 1e-9)
                cd = np.where((th > 1e-6)[:, None], (np.sin(th - al)[:, None] * cd + np.sin(al)[:, None] * td) / sn[:, None], cd)
                cm = cm + np.clip(fm - cm, -m_rate, m_rate)
                a = a + cm[:, None] * cd
            v = v + a * dt; p = p + v * dt; P[i + 1] = p
        return P

    def _coast_plan(self, p, v, obs, t_left, horizon=12.0, dt=0.1):
        """Burn-tail coast planner: for each candidate thrust (direction x magnitude) held for the burn left, predict
        the path (incl. coasting) and every live balloon's closest approach; P(hit) falls off with the miss distance
        beyond 1.2 m, with an error that grows with time (worse after burnout). -> (best force, expected hits,
        expected hits of every candidate)."""
        st, bs = obs["balloon_status"][:, 0], obs["balloon_states"]
        alive = np.flatnonzero(st == 1)
        if alive.size == 0:
            return None, 0.0, None
        mags = self.a_thrust * np.array([self.thr_lo, 0.5 * (self.thr_lo + self.thr_hi), self.thr_hi])
        F = np.vstack([(self.plan_dirs[None] * mags[:, None, None]).reshape(-1, 3), -G])  # + "hold altitude"
        K, w = len(F), self._wind(obs)
        f0 = quat_to_R(self.q)[:, 2] * self.throttle * self.a_thrust  # thrust now
        P = self._fly(np.tile(p, (K, 1)), np.tile(v, (K, 1)), F, t_left, w, t_left + horizon, dt, f0)
        b0, bv = bs[alive, :3], bs[alive, 3:]
        best = np.full((K, alive.size), np.inf); tbest = np.zeros((K, alive.size))
        live = np.ones(K, bool)
        for i in range(len(P) - 1):  # exact closest approach on each segment (relative motion ~ linear within dt)
            live &= P[i, :, 2] > self.elev
            r0 = (b0 + bv * i * dt)[None] - P[i][:, None]; r1 = (b0 + bv * (i + 1) * dt)[None] - P[i + 1][:, None]
            dr = r1 - r0
            s = np.clip(-(r0 * dr).sum(-1) / np.maximum((dr * dr).sum(-1), 1e-9), 0.0, 1.0)
            dist = np.linalg.norm(r0 + dr * s[..., None], axis=-1)
            dist[~live] = np.inf
            upd = dist < best
            best[upd] = dist[upd]; tbest[upd] = ((i + s) * dt)[upd]
        sig = 0.3 + 0.05 * tbest + 0.4 * np.maximum(tbest - t_left, 0.0)
        eh = np.exp(-0.5 * (np.maximum(best - 1.2, 0.0) / sig) ** 2).sum(1)
        k = int(np.argmax(eh))
        return F[k], float(eh[k]), eh

    def _guidance(self, p, v, bpos, bvel, t_go):
        """N * zero-effort-miss / t_go^2, gravity cancelled separately (N = 2: constant-thrust hit).
        coast: a hit after burnout needs force f = (miss incl. gravity) / _burn_disp(t_go) while burning."""
        t_go = max(t_go * self.stretch * self.s_mul, self.t_go_min)
        if self.bwind or self.aim_prof is not None:
            bpos = bpos + self._bdrift(bpos[2], bvel[2], t_go, self.aim_prof)
        if self.coast and t_go > self.burn_left:
            return 0.5 * self.nav_N * self.n_mul * (bpos - p + (bvel - v) * t_go - 0.5 * G * t_go**2) / self._burn_disp(t_go) + G
        return self.nav_N * self.n_mul * (bpos - p + (bvel - v) * t_go) / t_go**2

    def _arrv(self, t_go):
        """Arrival velocity offset dv: fly the deviation (dv_p, dv_v) that this term causes (from 0 at target acquisition) with
        ZEM/ZEV guidance to (0, dv) at arrival: a = 6 Z / t^2 - 2 (dv - dv_v) / t, Z = -(dv_p + dv_v t) its part of the ZEM; the
        ZEM term already gives N Z / t^2 (N = nav_N n_mul) -> the rest. Linear, so the flight = the dv = 0 one + that deviation:
        miss unchanged, arrival velocity + dv. Held dv does not add up; at acquisition it is -2 dv / t (= ZEM/ZEV with
        v_f = the N = 2 arrival velocity + dv). t = _guidance's t_go (floor 0.5 s), faded out between t 0.7 and 0.3 s,
        |.| <= vcap * a_thrust. Not when the hit is after burnout (coast branch)."""
        t = max(t_go * self.stretch * self.s_mul, self.t_go_min)
        if self.target != self.dv_tgt:
            self.dv_p, self.dv_v, self.dv_tgt = np.zeros(3), np.zeros(3), self.target
        if self.coast and t > self.burn_left:
            return np.zeros(3)
        N, tf = self.nav_N * self.n_mul, max(t, 0.5)
        z = -(self.dv_p + self.dv_v * t)
        da = ((6.0 - N) * z / tf - 2.0 * (self.dv - self.dv_v)) / tf * min(max((t - 0.3) / 0.4, 0.0), 1.0)
        n, cap = np.linalg.norm(da), self.vcap * self.a_thrust
        da = da * (cap / n) if n > cap else da
        a = N * z / t**2 + da  # the deviation's acceleration (the ZEM term's share + this)
        self.dv_p = self.dv_p + self.dv_v * self.dt + 0.5 * a * self.dt**2
        self.dv_v = self.dv_v + a * self.dt
        return da

    def _eob(self, p, v, obs, t_go, f):
        """End-of-burn mode: the policy's target cannot be hit before burnout (eob=1: t_go * stretch > burn left T;
        eob=2: see _eob_pre), so hit it tau in (0, eob_tau] s after burnout with a constant force over the rest of the burn:
        B(T + tau) = p + v (T + tau) + G (T + tau)^2 / 2 + f_net (T^2 / 2 + T tau), f = f_net - G; tau minimises |f|
        (capped at the thrust; no floor on f_z). The last eob_hold s hold the thrust. -> force (f if the mode is off)."""
        T = self.burn_left
        if self.eob_f is not None and T <= self.eob_hold:
            if self.eob_align:
                ang = np.degrees(np.arccos(np.clip(unit(self.eob_f) @ unit(v), -1.0, 1.0)))
                self._cnt("eob_hold"); self.opt_n["eob_hold_min_deg"] = min(self.opt_n.get("eob_hold_min_deg", 180), round(ang, 1))
                if ang < self.eob_align:
                    self._cnt("eob_align")
                    return np.linalg.norm(self.eob_f) * unit(v)
            return self.eob_f
        b = obs["balloon_states"][self.target] if self.target is not None else None
        if b is None or T > self.eob_T or (p[2] - self.elev < self.min_alt and v[2] < 0) or \
                (self.eob == 1 and t_go * self.stretch * self.s_mul <= T) or (self.eob >= 2 and self._eob_pre(p, v, b, T)):
            self.eob_f = None
            return f
        tau = np.arange(0.05, self.eob_tau + 1e-9, 0.05)
        tt = T + tau
        B = b[:3] + b[3:] * tt[:, None]
        if self.bwind:
            B = B + self._bdrift(b[2], b[5], tt, self.aim_prof)
        F = (B - p - v * tt[:, None] - 0.5 * G * tau[:, None] ** 2) / (0.5 * T**2 + T * tau)[:, None] - G
        k = int(np.argmin(np.linalg.norm(F, axis=1)))
        if self.eob_pick:
            k = self._eob_pick(obs, v, T, tau, B, F, k)
        fe = F[k] * min(1.0, self.a_thrust * self.thr_hi / max(np.linalg.norm(F[k]), 1e-9))
        self.eob_f, self.eob_tgt = fe, self.target
        self.debug_eob = (T, tau[k], np.linalg.norm(F[k]))
        return fe

    def _eob_pick(self, obs, v, T, tau, B, F, k, horizon=1.5, ds=0.05):
        """eob_pick: index of the min-force feasible tau whose ballistic post-hit path (from the hit point B, velocity
        v + (F + G) T + G tau) passes within eob_pick_r of another live balloon in horizon s; none -> k."""
        st, bs = obs["balloon_status"][:, 0], obs["balloon_states"]
        oth = np.flatnonzero((st == 1) & np.isfinite(bs).all(1)); oth = oth[oth != self.target]
        fn = np.linalg.norm(F, axis=1)
        feas = fn <= self.a_thrust * self.thr_hi
        self._cnt("eob_pick_eval")
        if not oth.size or not feas.any():
            return k
        s = np.arange(0.0, horizon + 1e-9, ds)[None, :, None]
        vh = v + (F + G) * T + G * tau[:, None]
        path = B[:, None] + vh[:, None] * s + 0.5 * G * s**2  # (K, S, 3)
        ts = (T + tau)[:, None, None] + s  # (K, S, 1)
        ob = bs[oth, :3] + bs[oth, 3:] * ts[..., None]  # (K, S, M, 3)
        if self.bwind:
            ob = ob + self._bdrift(bs[oth, 2], bs[oth, 5], ts, self.aim_prof)
        near = (np.linalg.norm(ob - path[:, :, None], axis=-1) <= self.eob_pick_r).any((1, 2)) & feas
        if not near.any():
            return k
        self._cnt("eob_pick_near")
        j = int(np.flatnonzero(near)[np.argmin(fn[near])])
        if j != k:
            self._cnt("eob_pick")
        return j

    def _cnt(self, key):
        self.opt_n[key] = self.opt_n.get(key, 0) + 1

    def _ho_blend(self, p, v, obs, a_new):
        """ho_keep: must-hit blend toward the released target ho_old (gate.py Fc: Gate._guidance / _blend / _t_hit)."""
        bs = obs["balloon_states"]
        ho = bs[self.ho_old].astype(float)
        to = self._reach_time(p, v, ho[None, :3], ho[None, 3:])[0]
        if not np.isfinite(to):
            self.ho_old = None; return a_new
        a_old = self._guidance(p, v, ho[:3], ho[3:], to)
        nb = bs[self.target].astype(float)
        hor = 1.5 * max(to * self.stretch * self.s_mul, self.t_go_min) + 0.5
        TH = np.arange(0.0, 6.0, 0.02)
        best = None
        for lam in np.linspace(0.0, 1.0, 11):
            a = (1 - lam) * a_old + lam * a_new
            net = self._saturate(a) + G
            rel = (ho[:3] - p)[None] + (ho[3:] - v)[None] * TH[:, None] - 0.5 * net[None] * TH[:, None] ** 2
            ok = (np.linalg.norm(rel, axis=1) < self.ho_keep) & (TH <= hor)
            if not ok.any():
                continue
            th = TH[int(np.argmax(ok))]
            ph, vh = p + v * th + 0.5 * net * th**2, v + net * th
            t2 = self._reach_time(ph, vh, (nb[:3] + nb[3:] * th)[None], nb[None, 3:])[0]
            c = th + t2 if np.isfinite(t2) else np.inf
            if best is None or c < best[0] or (c == best[0] and lam > best[1]):
                best = (c, lam, a)
        if best is None:
            self.ho_old = None; return a_new
        if best[1] < 1.0:
            self._cnt("ho_keep")
        return best[2]

    def _eob_pre(self, p, v, b, T):
        """eob=2: True when the target b can be hit by burnout with a constant force within the thrust (then the
        normal guidance flies it), unless the mode is already on for this target."""
        if self.eob_f is not None and self.eob_tgt == self.target:
            return False
        t = np.append(np.arange(0.05, T, 0.05), T)
        B = b[:3] + b[3:] * t[:, None]
        if self.bwind:
            B = B + self._bdrift(b[2], b[5], t, self.aim_prof)
        F = 2.0 * (B - p - v * t[:, None]) / (t**2)[:, None] - G
        return bool((np.linalg.norm(F, axis=1) <= self.a_thrust * self.thr_hi).any())

    def _saturate(self, a_cmd):
        """Specific force for a_cmd, scaled so gravity compensation always fits in a_thrust."""
        g = -G
        a = a_cmd.copy()
        a[2] = max(a[2], 1.0 - g[2])  # never thrust downward
        # largest s in [0, 1] with |s a + g| <= a_thrust
        A, B, C = a @ a, 2 * a @ g, g @ g - self.a_thrust**2
        s = 1.0
        if A > 1e-9 and (a + g) @ (a + g) > self.a_thrust**2:
            s = max((-B + np.sqrt(max(B * B - 4 * A * C, 0.0))) / (2 * A), 0.0)
        return s * a + g

    def _spend_excess(self, f, v):
        """Throttling down does not save propellant, so put unused thrust along `excess_dir`
        (0: throttle down, 1: straight up = store altitude, 2: along velocity = speed up, 3: along the commanded net
        acceleration = toward the intercept, harder; 4: toward the next target, self.excess_to: the policy's second
        choice / the lookahead's next balloon, set in get_action; DECISION_LOG §199)."""
        mode = int(self.excess_dir)
        if mode == 0:
            return f
        if mode == 3:
            d = unit(f + G)
        elif mode == 4:
            d = getattr(self, "excess_to", None)
            if d is None:
                return f
        else:
            d = np.array([0.0, 0.0, 1.0]) if mode == 1 else unit(v)
        if not d.any():
            return f
        # largest k >= 0 with |f + k d| = a_thrust
        b, c = f @ d, f @ f - self.a_thrust**2
        k = -b + np.sqrt(max(b * b - c, 0.0))
        return f + max(k, 0.0) * self.excess_frac * d

    def _attitude_control(self, f_des, gyro):
        """TVC and roll command to point the nose (body z) along f_des."""
        R = quat_to_R(self.q)
        d_body = R.T @ unit(f_des)
        err = np.cross([0.0, 0.0, 1.0], d_body)  # rotation axis * sin(angle)
        ang = np.arctan2(np.linalg.norm(err), d_body[2])
        w_cmd = self.k_att * ang * unit(err)
        w_cmd = w_cmd * min(1.0, self.max_rate / max(np.linalg.norm(w_cmd), 1e-9))
        tvc = self.k_rate * (w_cmd[:2] - gyro[:2])
        tvc *= min(1.0, self.max_gimbal / max(np.linalg.norm(tvc), 1e-9))
        roll = np.clip(-self.k_roll * gyro[2], -self.max_roll, self.max_roll)
        return tvc, roll

    # ------------------------------------------------------------------ main
    def get_action(self, obs):
        if self.bwind >= 4 or self.dp_hist:  # the history profile collects from the first step (also on the pad)
            self.hstep += 1
            if self.hstep % 10 == 1:
                self._hist_add(obs)
        action = self._gnc_action(obs)
        if self.lag_comp and self.launched_at is not None:  # flying: the environment applies this step's commands
            want = np.array([*action["tvc"], action["roll"], action["throttle"]], float)
            y, a = self.act_out, self.act_a
            u = np.clip(y + (np.clip(want, y - self.act_rate, y + self.act_rate) - y) / a, self.act_lo, self.act_hi)
            self.act_out = np.clip(np.clip(a * u + (1 - a) * y, y - self.act_rate, y + self.act_rate), self.act_lo, self.act_hi)
            action.update(tvc=u[:2], roll=float(u[2]), throttle=float(u[3]))
        return action

    def _gnc_action(self, obs):
        t = float(obs["simulation_time"])
        sens = obs["rocket_sensors"]
        action = {
            "launch": t >= self.launch_time,
            "launch_inclination_heading": self.incl_head,
            "tvc": np.zeros(2),
            "roll": 0.0,
            "throttle": 1.0,
        }
        if np.isnan(sens).any():
            return action
        if self.launched_at is None:
            self.launched_at = t - self.dt

        gyro, acc = sens[:3], sens[3:6]
        self.q = quat_step(self.q, gyro, self.dt)
        p, v = self._navigate(acc, sens[6:9], sens[9:12])

        if not self._burning(t):
            self.debug = {"t": t, "target": None, "q": self.q.copy()}
            return action

        # refine thrust acceleration estimate (axial specific force / throttle)
        thr = self.act_out[3] if self.lag_comp else self.throttle  # lag_comp: the throttle actually applied
        if thr > 0.3 and acc[2] > 1.0:
            self.a_thrust += 0.05 * (acc[2] / thr - self.a_thrust)

        self.burn_left = max(self.launched_at + self.burn_time - t, self.dt)
        self.step_i = getattr(self, "step_i", 0) + 1
        if 0 < self.bwind < 4 and self.step_i % 5 == 1:
            self.wprof = self._wind_profile(obs)
        self.target, t_go = self._select_target(p, v, obs)
        self.t_go_last = (self.target, t_go)  # (rl2 ho_T trigger)
        if self.v2t > 0:
            self.dv = np.zeros(3)
            if self.target is not None and self.next2 is not None:
                ap, av, p2 = self.next2; u = p2 - ap; n = np.linalg.norm(u)
                if n > 1e-6:
                    dv = self.v2t * (np.linalg.norm(av) * u / n - av); m = np.linalg.norm(dv)
                    self.dv = dv * (self.v2t_max / m) if m > self.v2t_max else dv
        if self.ho_old is not None and (obs["balloon_status"][self.ho_old, 0] != 1 or self.ho_old == self.target):
            self.ho_old = None
        if self.target is None:
            a_cmd = np.zeros(3)
        else:
            b = obs["balloon_states"][self.target]
            a_cmd = self._guidance(p, v, b[:3], b[3:], t_go)
            if self.ho_old is not None:  # (eob, when on, replaces f below: it wins in its window)
                a_cmd = self._ho_blend(p, v, obs, a_cmd)
            if self.res.any():  # (eob / coast planner replace f below, so it never reaches them)
                a_cmd = a_cmd + self.res
            if self.dv.any() or self.dv_v.any() or self.dv_p.any():  # (same gating)
                a_cmd = a_cmd + self._arrv(t_go)

        # ground avoidance: never plan to fall below min_alt
        alt = p[2] - self.elev
        if alt < self.min_alt and v[2] < 0:
            a_cmd[2] = max(a_cmd[2], v[2] ** 2 / (2 * max(alt, 1.0)))

        f = self._saturate(a_cmd)
        if int(self.excess_dir) == 4:  # direction to the next target: the policy's second choice (RL3 tc2), else the lookahead's
            j2 = getattr(self, "j2", None)
            q = obs["balloon_states"][j2][:3] if j2 is not None and obs["balloon_status"][j2, 0] == 1 else (
                self.next2[2] if getattr(self, "next2", None) is not None else None)
            self.excess_to = None if q is None else unit(np.asarray(q, float) - p)
        f = self._spend_excess(f, v)
        if self.eob:
            f = self._eob(p, v, obs, t_go, f)
        if self.coast >= 2 and self.burn_left <= self.plan_T:
            # the target cannot be hit before burnout (or there is none): the planner may take over for the rest
            if not self.plan_on and (self.target is None or t_go * self.stretch * self.s_mul > self.burn_left):
                if self.plan is None or self.step_i - self.plan[2] >= self.plan_every:
                    fp, eh, _ = self._coast_plan(p, v, obs, self.burn_left)
                    self.plan = (fp, eh, self.step_i)
                    self.plan_on = fp is not None and eh >= self.plan_min
            if self.plan_on:
                f = self.plan[0].copy()
                if self.burn_left <= self.align_T:  # along the velocity: no angle of attack at burnout
                    f = np.linalg.norm(f) * unit(v)
        self.f_gnc = f  # (hybrid E2E imitation target, dev/e2e.py)
        if getattr(self, "f_ext", None) is not None:  # hybrid E2E (DECISION_LOG §249): the policy's thrust demand replaces the
            f = self.f_ext                             # guidance's; throttle rate limit and attitude loop as before
        if getattr(self, "f_res", None) is not None:  # hybrid E2E residual (§262): a policy's correction added to the guidance's
            f = f + self.f_res
        thr_cmd = float(np.clip(np.linalg.norm(f) / self.a_thrust, self.thr_lo, self.thr_hi))
        step = self.thr_rate * self.dt
        self.throttle = float(np.clip(thr_cmd, self.throttle - step, self.throttle + step))

        tvc, roll = self._attitude_control(f, gyro)
        action.update(tvc=tvc, roll=float(roll), throttle=self.throttle)
        self.debug = {"t": t, "target": self.target, "q": self.q.copy(), "f": f}
        return action

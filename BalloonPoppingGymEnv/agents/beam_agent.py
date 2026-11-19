"""Beam-search route planner on top of GNCAgent's navigation and attitude control.

Leg model (balloon moves at constant velocity, drag ignored): from state (p, v) the rocket
hits a balloon t seconds later with a constant specific force f held until burnout
(tb seconds from leg start), then coasting:

    rel(t) = r + (vb - v) t - G t^2 / 2 = f * w(t),   w(t) = t^2/2 (t <= tb), tb t - tb^2/2 (t > tb)

so a leg of duration t is feasible iff |rel(t)| <= (a_thrust - margin) * w(t). Legs may end
after burnout, which plans the final coast into a balloon.

Uses only observation and given_parameters.
"""

import numpy as np

from BalloonPoppingGymEnv.agents.gnc_agent import G, GNCAgent, quat_step, quat_to_R
from BalloonPoppingGymEnv.agents.leg_model import LegModel


class BeamAgent(GNCAgent):
    def __init__(self, given_parameters, **kw):
        super().__init__(given_parameters, **kw)
        self.depth = int(kw.get("depth", 4))
        self.beam = int(kw.get("beam", 8))
        self.branch = int(kw.get("branch", 5))  # nearest targets expanded per state
        self.stretches = np.array(kw.get("stretches", [1.0, 1.25, 1.6]))
        self.replan_every = kw.get("replan_every", 0.25)  # (s)
        self.freeze_t = kw.get("freeze_t", 0.25)  # (s) hold command in the last moments
        self.min_hit_alt = kw.get("min_hit_alt", 8.0)  # (m AGL) ignore balloons lower than this
        self.min_path_alt = kw.get("min_path_alt", 3.0)  # (m AGL) legs must stay above
        self.keep_bias = kw.get("keep_bias", 0.3)  # (s) prefer the current first target
        self.leg_grid = np.arange(0.05, kw.get("max_leg", 12.0), kw.get("leg_dt", 0.05))
        # learned reachability (dev/train_legs.py) replaces the drag-free analytic feasibility test
        self.leg_model = LegModel() if kw.get("learned_legs", 1) else None
        self.p_hit = kw.get("p_hit", 0.5)  # min predicted hit probability for a feasible leg
        # borrowed from GNCAgent, off by default
        self.beam_N = kw.get("beam_N", 0.0)  # >0: ZEM gain N on the planned arrival time (2 == _fly while burning)
        self.ground_alt = kw.get("ground_alt", 0.0)  # (m AGL) GNC-style pull-up floor while descending
        self.commit_margin = kw.get("commit_margin", 0.3)  # (m/s^2) reserve to keep a commitment
        self.plan = None  # (target index, absolute arrival time)
        self.next_replan = -1.0

    # ------------------------------------------------------------------ model
    def _w(self, t, tb):
        tb = np.minimum(tb, t)
        return tb * t - 0.5 * tb**2

    def _legs(self, P, V, tb, BP, BV, Z):
        """Min feasible leg time from each state to each balloon.

        P, V, Z (S,3) position, velocity, thrust axis; tb (S,); BP, BV (S,N,3) balloon states
        at the leg start -> (S,N)."""
        tg = self.leg_grid[None, None, :, None]
        if self.leg_model is not None:
            S, N = BP.shape[:2]
            T = len(self.leg_grid)
            x = np.empty((S, N, T, 14))
            x[..., 0:3] = (BP - P[:, None])[:, :, None]
            x[..., 3:6] = (BV - V[:, None])[:, :, None]
            x[..., 6:9] = V[:, None, None]
            x[..., 9:12] = Z[:, None, None]
            x[..., 12] = tb[:, None, None]
            x[..., 13] = self.leg_grid[None, None, :]
            ok = self.leg_model(x) >= self.p_hit
        else:
            rel = (BP - P[:, None])[:, :, None] + (BV - V[:, None])[:, :, None] * tg - 0.5 * G * tg**2
            w = self._w(self.leg_grid[None, :], tb[:, None])[:, None, :]  # (S,1,T)
            ok = np.linalg.norm(rel, axis=-1) <= (self.a_thrust - self.margin) * w
        first = np.argmax(ok, axis=-1)
        return np.where(ok.any(axis=-1), self.leg_grid[first], np.inf)

    def _fly(self, p, v, bp, bv, t, tb):
        """Specific force and arrival velocity for a leg of duration t (vectorized)."""
        t = t[..., None]
        rel = bp - p + (bv - v) * t - 0.5 * G * t**2
        w = self._w(t, tb[..., None])
        f = rel / np.maximum(w, 1e-9)
        tbc = np.minimum(tb[..., None], t)
        v_end = v + G * t + f * tbc
        return f, v_end

    def _path_ok(self, p, v, f, t, tb):
        """Leg stays above min_path_alt (checked at a few points)."""
        ok = np.ones(t.shape, bool)
        for s in (0.25, 0.5, 0.75):
            ts = t * s
            tbc = np.minimum(tb, ts)
            z = p[..., 2] + v[..., 2] * ts - 4.905 * ts**2 + f[..., 2] * (tbc * ts - 0.5 * tbc**2)
            ok &= z >= self.elev + self.min_path_alt
        return ok

    # ------------------------------------------------------------------ planner
    def _plan(self, t_now, p, v, obs):
        status = obs["balloon_status"][:, 0]
        bstate = obs["balloon_states"]
        alive = np.flatnonzero(status == 1)
        if alive.size == 0:
            return None
        bp0, bv0 = bstate[alive, :3], bstate[alive, 3:]
        n = alive.size
        burn_left = self.launched_at + self.burn_time - t_now

        P, V = p[None].copy(), v[None].copy()
        Z = quat_to_R(self.q)[:, 2][None]
        Tc = np.zeros(1)
        used = np.zeros((1, n), bool)
        first = np.full(1, -1)
        first_t = np.zeros(1)
        best = None  # (depth, time, first, first_t)

        for d in range(self.depth):
            S = P.shape[0]
            BP = bp0[None] + bv0[None] * Tc[:, None, None]
            BV = np.broadcast_to(bv0[None], BP.shape)
            tb = np.maximum(burn_left - Tc, 0.0)
            tmin = self._legs(P, V, tb, BP, BV, Z)
            tmin[used] = np.inf
            tmin[BP[..., 2] - self.elev < self.min_hit_alt] = np.inf
            k = min(self.branch, n)
            idx = np.argsort(tmin, axis=1)[:, :k]  # (S,k)
            tk = np.take_along_axis(tmin, idx, axis=1)
            # expand: state x target x stretch
            si, ki, xi = np.meshgrid(np.arange(S), np.arange(k), np.arange(len(self.stretches)), indexing="ij")
            si, ki, xi = si.ravel(), ki.ravel(), xi.ravel()
            tgt = idx[si, ki]
            tleg = tk[si, ki] * self.stretches[xi]
            valid = np.isfinite(tleg)
            # stretched legs only while burning (after burnout the leg time is fixed by physics)
            valid &= (xi == 0) | (tleg <= tb[si])
            if not valid.any():
                break
            si, tgt, tleg = si[valid], tgt[valid], tleg[valid]
            bpt, bvt = BP[si, tgt], bv0[tgt]
            f, v_end = self._fly(P[si], V[si], bpt, bvt, tleg, tb[si])
            keep = self._path_ok(P[si], V[si], f, tleg, tb[si])
            if not keep.any():
                break
            si, tgt, tleg, bpt, bvt, v_end, f = si[keep], tgt[keep], tleg[keep], bpt[keep], bvt[keep], v_end[keep], f[keep]
            Tn = Tc[si] + tleg
            order = np.argsort(Tn)[: self.beam]
            si, tgt, tleg = si[order], tgt[order], tleg[order]
            P = bpt[order] + bvt[order] * tleg[:, None]
            V = v_end[order]
            Z = f[order] / np.maximum(np.linalg.norm(f[order], axis=-1, keepdims=True), 1e-9)  # next leg starts pointed along f
            used = used[si].copy()
            used[np.arange(len(si)), tgt] = True
            first = first[si] if d else tgt
            first_t = first_t[si] if d else tleg
            Tc = Tn[order]
            # prefer plans whose first target is the current one
            bias = np.zeros_like(Tc)
            if self.plan is not None:
                bias[alive[first] == self.plan[0]] = self.keep_bias
            j = np.argmin(Tc - bias)
            best = (d, Tc[j], alive[first[j]], first_t[j])
        if best is None:
            return None
        tgt, t_arr = best[2], t_now + best[3]
        if self.plan is not None and self.plan[0] == tgt and self.plan[1] - t_now > self.freeze_t:
            # same target: keep the committed arrival time while it is still flyable, so the
            # force command does not jump at every replan
            b = bstate[tgt]
            tb = np.array(burn_left)
            f, _ = self._fly(p, v, b[:3], b[3:], np.array(self.plan[1] - t_now), tb)
            if np.linalg.norm(f) <= self.a_thrust - self.commit_margin:
                t_arr = self.plan[1]
        return tgt, t_arr

    # ------------------------------------------------------------------ main
    def get_action(self, obs):
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
            self.debug = {"t": t, "target": None}
            return action
        if self.throttle > 0.3 and acc[2] > 1.0:
            self.a_thrust += 0.05 * (acc[2] / self.throttle - self.a_thrust)

        status = obs["balloon_status"][:, 0]
        if (
            self.plan is None
            or t >= self.next_replan
            or status[self.plan[0]] != 1
            or t > self.plan[1]
        ):
            self.plan = self._plan(t, p, v, obs)
            self.next_replan = t + self.replan_every
        self.target = None if self.plan is None else self.plan[0]

        if self.plan is None:
            a = np.zeros(3)  # hover
        else:
            tgt, t_arr = self.plan
            t_go = max(t_arr - t, self.freeze_t)
            b = obs["balloon_states"][tgt]
            burn_left = self.launched_at + self.burn_time - t
            if self.beam_N > 0 and t_go <= burn_left:
                a = self.beam_N * (b[:3] - p + (b[3:] - v) * t_go) / t_go**2
            else:
                f, _ = self._fly(p, v, b[:3], b[3:], np.array(t_go), np.array(burn_left))
                a = f + G  # back to acceleration for the shared limiter
        alt = p[2] - self.elev
        if alt < self.ground_alt and v[2] < 0:
            a[2] = max(a[2], v[2] ** 2 / (2 * max(alt, 1.0)))
        f = self._spend_excess(self._saturate(a), v)  # excess_dir=0 (default) is a no-op
        thr_cmd = float(np.clip(np.linalg.norm(f) / self.a_thrust, self.thr_lo, self.thr_hi))
        step = self.thr_rate * self.dt
        self.throttle = float(np.clip(thr_cmd, self.throttle - step, self.throttle + step))
        tvc, roll = self._attitude_control(f, gyro)
        action.update(tvc=tvc, roll=float(roll), throttle=self.throttle)
        self.debug = {"t": t, "target": self.target, "f": f}
        return action


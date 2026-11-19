"""hist: wind-vs-altitude profile accumulated over the WHOLE observation history, balloons drift with it as they rise.

Facts checked on TRAIN seeds (data/traj/s4, seeds 20000-20031):
  * a balloon's horizontal velocity is a function of altitude only (spread < 0.02 m/s among all balloons that pass the
    same 0.5 m altitude bin at any time) and settles within ~0.2 s of release; its vertical speed is constant per balloon
    (5.8..6.4 m/s) after the same transient -> every past (altitude, velocity) sample of any balloon samples one profile
  * that profile is a C2 cubic spline with knots every 20 m of altitude (value, slope and curvature continuous at
    elev + 20 j, third derivative jumps) around a mean wind of ~(-1.3, 1.3) m/s, gust amplitude shrinking with altitude.
Profile: every sample is hat-binned on a 0.5 m altitude grid as local-linear moments (running sums, only new rows are
added), so each node's value is unbiased even where the data are one-sided (the top of the profile); a node with a
single sample (only the top balloon has passed it) borrows the slope of the nearest node below. The accuracy of the
top few metres matters most: they drive the extrapolation.
Above the highest balloon ever seen: a ridge-regression map fitted on TRAIN profiles (fit), one per 0.25 m phase of the
top within its 20 m spline segment (blended between neighbouring phases), takes the profile's deviation from the mean
wind at top - lags (up to 60 m below) to the deviation up to 45 m above the top, both scaled by the gust amplitude at the
top. It learns the exact spline continuation of the partly seen segment and the mean-reverting prior beyond it (the
knot values are independent, uniform-looking, so the next segments are genuinely unknown).
Prediction integrates the profile along each balloon's rise at its own vertical speed (the median of the others if it
was released < a_min ago), exactly like GNCAgent._bdrift.
Hyperparameters (dz, a_min, nph, lags, ridge) were chosen by 2-fold cross-validation on the TRAIN seeds only.
Agent use: call update(...) with the newest rows every step (or every few steps: ~0.4 ms incl. the profile rebuild), then
predict(...) (~0.1 ms for 100 balloons x 6 horizons); a time going backwards resets the history (new episode).
"""
import numpy as np


class Predictor:
    def __init__(self, given, dz=0.5, a_min=0.25, knot=20.0, nph=80, lags=np.r_[0:4:0.5, 4:21:2, 24:61:4],
                 dext=np.arange(0.5, 45.1, 0.5), ridge=1e-6, extrap=True):
        self.elev, self.nb = given["environment"]["elevation"], int(given["balloon"]["num"])
        self.dz, self.a_min, self.knot, self.nph, self.extrap = dz, a_min, knot, nph, extrap
        self.z = self.elev + np.arange(0.0, 1200.0, dz)  # (m ASL) profile grid (balloons rise ~6.1 m/s for <= 150 s)
        self.lags, self.dext, self.ridge = np.asarray(lags, float), np.asarray(dext, float), ridge
        self.m, self.s_ab, self.A = np.zeros(2), (0.0, 0.0), None  # mean wind, gust scale exp(a + b z_agl), maps
        self.reset()

    def reset(self):
        self.t_last, self.rel = -np.inf, np.full(self.nb, np.nan)
        self.mom = np.zeros((5, len(self.z), 2))  # per bin: sum w, w d, w d^2 (same in both columns), w v, w d v
        self.ztop, self.W, self.Wc, self.vz_med = -np.inf, None, None, 6.1

    # ------------------------------------------------------------------ learned extrapolation (TRAIN only)
    def _scale(self, z):
        return np.exp(self.s_ab[0] + self.s_ab[1] * (z - self.elev))

    def _vals(self, W, zq):
        return np.stack([np.interp(zq, self.z, W[:, k]) for k in (0, 1)], -1)

    def _feat(self, W, zt):
        """normalised deviations of the profile at zt - lags -> (..., 2, L) (one row per wind component)."""
        zt = np.asarray(zt, float)
        v = self._vals(W, np.maximum(zt[..., None] - self.lags, self.elev + 1.5))
        return np.swapaxes((v - self.m) / self._scale(zt)[..., None, None], -1, -2)

    def _map(self, zt):
        """extrapolation map for a top at zt: linear blend of the two nearest phase-bin maps (circular)."""
        f = ((zt - self.elev) % self.knot) / self.knot * self.nph - 0.5
        b = int(np.floor(f)); f -= b
        return (1 - f) * self.A[b % self.nph] + f * self.A[(b + 1) % self.nph]

    def fit(self, data, step=0.125):
        profs = []
        for d in data:  # full-episode profile of each TRAIN seed (all settled samples, sorted by altitude)
            S, St = d["states"], d["status"]
            age = np.arange(len(St))[:, None] - (St == 1).argmax(0)[None]
            fly = (St == 1) & (age >= 3) & (S[:, :, 2] > self.elev + 1.5)
            z, v = S[:, :, 2][fly], S[:, :, 3:5][fly]
            o = np.argsort(z)
            profs.append((z[o[-1]], np.stack([np.interp(self.z, z[o], v[o, k]) for k in (0, 1)], 1)))
        inside = [(self.z >= self.elev + 1.5) & (self.z <= zm) for zm, _ in profs]
        allv = np.concatenate([W[m] for (_, W), m in zip(profs, inside)])
        self.m = allv.mean(0)
        zc = np.concatenate([self.z[m] for m in inside]) - self.elev  # gust scale: log RMS deviation per 20 m band
        band, dev2 = np.floor(zc / 20.0), ((allv - self.m) ** 2).mean(1)
        ub = np.unique(band)
        self.s_ab = tuple(np.polyfit(ub * 20.0 + 10.0, np.log([np.sqrt(dev2[band == u].mean()) for u in ub]), 1)[::-1])
        L, D = len(self.lags), len(self.dext)
        xx, xy, cnt = np.zeros((self.nph, L, L)), np.zeros((self.nph, L, D)), np.zeros(self.nph)
        for zm, W in profs:  # every "top" altitude of every seed, grouped by its phase in the 20 m spline segment
            zt = np.arange(self.elev + 10.0, zm - self.dext[-1], step)
            x = self._feat(W, zt)  # (n, 2, L)
            y = np.swapaxes((self._vals(W, zt[:, None] + self.dext) - self.m) / self._scale(zt)[:, None, None], 1, 2)
            ph = (((zt - self.elev) % self.knot) / self.knot * self.nph).astype(int) % self.nph
            for b in np.unique(ph):  # both wind components pooled
                xb, yb = x[ph == b].reshape(-1, L), y[ph == b].reshape(-1, D)
                xx[b] += xb.T @ xb; xy[b] += xb.T @ yb; cnt[b] += len(xb)
        self.A = [np.linalg.solve(xx[b] + self.ridge * cnt[b] * np.eye(L), xy[b]) for b in range(self.nph)]

    # ------------------------------------------------------------------ profile from the observation history
    def update(self, states, status, t):
        """Add the rows newer than the last call (states / status / t may be the whole history or only the newest
        rows); a time going backwards starts a new episode."""
        if t[-1] < self.t_last:
            self.reset()
        r0 = int(np.searchsorted(t, self.t_last, "right"))
        if r0 == len(t):
            return
        self.t_last = float(t[-1])
        S, on = states[r0:], status[r0:] == 1
        new = np.isnan(self.rel) & on.any(0)
        self.rel[new] = t[r0 + on[:, new].argmax(0)]
        age = t[r0:, None] - self.rel[None]
        mature = on[-1] & (age[-1] >= self.a_min)
        if mature.sum() >= 3:
            self.vz_med = float(np.median(states[-1, mature, 5]))
        ok = on & (age >= self.a_min) & (S[:, :, 2] > self.elev + 1.0)
        if not ok.any():
            return
        z, v = S[:, :, 2][ok].astype(float), S[:, :, 3:5][ok].astype(float)
        p = np.clip((z - self.z[0]) / self.dz, 0, len(self.z) - 1.001)
        i = p.astype(int)
        for j in (i, i + 1):  # hat weights to the two neighbouring grid nodes, moments about each node
            dd = z - self.z[j]
            w = 1.0 - np.abs(dd) / self.dz
            for c, q in enumerate((w, w * dd, w * dd * dd)):
                self.mom[c] += np.bincount(j, q, len(self.z))[:, None]
            self.mom[3] += np.stack([np.bincount(j, w * v[:, k], len(self.z)) for k in (0, 1)], 1)
            self.mom[4] += np.stack([np.bincount(j, w * dd * v[:, k], len(self.z)) for k in (0, 1)], 1)
        self.ztop = max(self.ztop, float(z.max()))
        self._profile()

    def _profile(self):
        s0, s1, s2, t0, t1 = self.mom
        ok = s0[:, 0] > 1e-6
        if ok.sum() < 2:
            return
        det = s0 * s2 - s1 * s1
        lin = (det > 1e-4 * s0 * s0 * self.dz ** 2)[:, 0]  # enough altitude spread in the bin for a local line
        slope = (s0 * t1 - s1 * t0) / np.where(det > 0, det, 1.0)
        near = np.maximum.accumulate(np.where(lin, np.arange(len(lin)), 0))  # else: slope of the nearest line below
        val = (t0 - slope[near] * s1) / np.maximum(s0, 1e-12)
        W = np.stack([np.interp(self.z, self.z[ok], val[ok, k]) for k in (0, 1)], 1)  # holds beyond the data
        zt = min(self.ztop, self.z[ok][-1])
        if self.extrap and self.A is not None:
            wt = self._vals(W, zt)
            zd = zt + self.dext
            vext = self.m[:, None] + (self._feat(W, zt) @ self._map(zt)) * self._scale(zt)  # (2, D)
            up = self.z > zt
            W[up] = np.stack([np.interp(self.z[up], np.r_[zt, zd], np.r_[wt[k], vext[k]]) for k in (0, 1)], 1)
        self.W = W
        self.Wc = np.vstack([np.zeros((1, 2)), np.cumsum(0.5 * (W[1:] + W[:-1]) * self.dz, 0)])

    # ------------------------------------------------------------------ prediction
    def predict(self, states, status, t, idx, horizons):
        self.update(states, status, t)
        s = states[-1, idx].astype(float)
        h = np.asarray(horizons, float)[None, :]
        vz = np.where(t[-1] - self.rel[idx] >= self.a_min, s[:, 5], self.vz_med)[:, None]
        z0 = s[:, 2:3]
        out = np.empty((len(idx), h.shape[1], 3))
        out[..., 2] = z0 + vz * h
        if self.W is None:
            out[..., :2] = s[:, None, :2] + s[:, None, 3:5] * h[..., None]
            return out
        rising = np.abs(vz) > 0.2
        for k in (0, 1):
            dW = np.interp(out[..., 2], self.z, self.Wc[:, k]) - np.interp(z0, self.z, self.Wc[:, k])
            out[..., k] = s[:, k:k + 1] + np.where(rising, dW / np.where(rising, vz, 1.0), s[:, 3 + k:4 + k] * h)
        return out

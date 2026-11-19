"""Spline + GP wind-profile predictor (scenario 4).

The horizontal wind depends on altitude only (verified on TRAIN: within 0.5 m altitude bins the velocities of different
balloons / times / places agree to ~0.02 m/s) and balloons move with it, so EVERY past balloon sample (z, vx, vy) is a
sample of one profile V(z).
 * observed altitudes: penalised regression spline (linear B-splines on a 0.5 m knot grid, 2nd-difference penalty lam)
   fitted to all past samples; the normal equations are accumulated incrementally (sample weight = its time step, so the
   fit does not depend on the sampling rate) and solved as one banded system.
 * above the highest observed altitude zmax: Gaussian-process posterior mean conditioned on the fitted profile (every
   `cstep` m over the top `cwin` m, nugget `noise`); kernel = c^2 (the seed's own mean wind)
   + s(z)s(z') exp(-d^2/2 ell^2) cos(2 pi d/per), s(z) = s0 exp(-(z - elev)/L). Right above zmax it continues the top
   balloon's own velocity and local trend; where the posterior variance grows it relaxes to the mean wind (`shrink` < 1
   would instead pull the extrapolation toward holding the top balloon's velocity; CV on TRAIN chose 1 = pure GP mean).
 * balloons rise at their current vertical speed vz and drift with V(z): x(h) = x0 + (C(z0 + vz h) - C(z0)) / vz,
   C = altitude integral of V (constant velocity if |vz| < 0.2 m/s).
Hyperparameters: cross-validation on the TRAIN seeds only (two TRAIN halves agreed); fit() re-estimates the prior mean
wind from TRAIN (default = that value). Uses only the observations passed so far and the given elevation.
"""
import numpy as np
from scipy.linalg import solveh_banded


class Predictor:
    HP = dict(dz=0.5, lam=0.1, zmin=1.0, ell=26.0, per=90.0, s0=2.8, L=400.0, c=0.2, noise=0.01, cstep=4.0,
              zext=120.0, cwin=120.0, shrink=1.0)

    def __init__(self, given, **hp):
        self.hp = {**self.HP, **hp}
        self.elev = given["environment"]["elevation"]
        self.zk = self.elev + np.arange(0.0, 1500.0, self.hp["dz"])  # (m ASL) spline knots
        self.mu = np.array([-1.33, 1.24])  # prior mean wind (m/s) = fit() on the TRAIN seeds
        self._reset(None)

    def fit(self, train):
        """Prior mean wind = mean balloon velocity over all TRAIN samples."""
        v = [d["states"][..., 3:5][(d["status"] == 1) & (d["states"][..., 2] > self.elev + 1.0)] for d in train]
        self.mu = np.concatenate(v).mean(0)

    # ---------------------------------------------------------------- profile
    def _reset(self, first):
        n = len(self.zk)
        self.first, self.nrow, self.zmax, self.prof = first, 0, -np.inf, None
        self.dg, self.of, self.by = np.zeros(n), np.zeros(n), np.zeros((n, 2))  # B'WB diagonal / superdiagonal, B'Wy

    def add(self, S, St, dt):
        """Accumulate observation rows S (r, N, 6), status St (r, N), time steps dt (r,) into the normal equations."""
        m = (St == 1) & (S[..., 2] > self.elev + self.hp["zmin"])
        if not m.any():
            return
        z, v, w = S[m][:, 2], S[m][:, 3:5], np.broadcast_to(np.asarray(dt, float)[:, None], m.shape)[m]
        n = len(self.zk)
        u = (z - self.zk[0]) / self.hp["dz"]
        i = np.clip(u.astype(int), 0, n - 2)
        f = u - i
        self.dg += np.bincount(i, w * (1 - f) ** 2, n) + np.bincount(i + 1, w * f * f, n)
        self.of += np.bincount(i, w * f * (1 - f), n)
        for k in (0, 1):
            self.by[:, k] += np.bincount(i, w * (1 - f) * v[:, k], n) + np.bincount(i + 1, w * f * v[:, k], n)
        self.zmax = max(self.zmax, z.max())

    def update(self, states, status, t):
        """Ingest the rows of the full history (states[:k+1], ...) not seen yet (restart on a new episode), refit."""
        if len(t) < self.nrow or self.first is None or not np.array_equal(self.first, states[0]):
            self._reset(states[0].copy())
        if len(t) == self.nrow:
            return
        dt = np.diff(t, prepend=t[0] - (t[1] - t[0] if len(t) > 1 else 0.01))
        self.add(states[self.nrow :], status[self.nrow :], dt[self.nrow :])
        self.nrow = len(t)
        if np.isfinite(self.zmax):
            self.prof = self._profile()

    def _spline(self):
        """P-spline values on the knots up to the first knot above zmax -> (knot altitudes (m,), V (m, 2))."""
        m = min(int((self.zmax - self.zk[0]) / self.hp["dz"]) + 2, len(self.zk))
        lam = self.hp["lam"]
        p0 = np.full(m, 6.0); p0[[0, -1]] = 1.0; p0[[1, -2]] = 5.0  # D2'D2 bands
        p1 = np.full(m - 1, -4.0); p1[[0, -1]] = -2.0
        ab = np.zeros((3, m))  # upper banded form of B'WB + lam D2'D2
        ab[2] = self.dg[:m] + lam * p0 + 1e-9
        ab[1, 1:] = self.of[: m - 1] + lam * p1
        ab[0, 2:] = lam
        return self.zk[:m], solveh_banded(ab, self.by[:m])

    def _kern(self, za, zb):
        h = self.hp
        sa, sb = (h["s0"] * np.exp(-(z - self.elev) / h["L"]) for z in (za, zb))
        d = za[:, None] - zb[None]
        rho = np.exp(-0.5 * (d / h["ell"]) ** 2) * np.cos(2 * np.pi * d / h["per"])
        return h["c"] ** 2 + sa[:, None] * sb[None] * rho

    def _extrap(self, zs, Vs, ze):
        """GP posterior mean of the wind at altitudes ze above zmax, given the fitted profile (zs, Vs)."""
        h = self.hp
        zc = self.zmax - np.arange(0.0, min(self.zmax - zs[0], h["cwin"]), h["cstep"])  # conditioning altitudes
        Vc = np.stack([np.interp(zc, zs, Vs[:, k]) for k in (0, 1)], 1)
        Lc = np.linalg.cholesky(self._kern(zc, zc) + h["noise"] ** 2 * np.eye(len(zc)))
        gp = self.mu + self._kern(ze, zc) @ np.linalg.solve(Lc.T, np.linalg.solve(Lc, Vc - self.mu))
        return Vc[0] + h["shrink"] * (gp - Vc[0])

    def _profile(self):
        """Wind on the whole knot grid (spline where observed, GP above, held beyond zext) and its altitude integral."""
        zs, Vs = self._spline()
        top = np.searchsorted(self.zk, self.zmax, side="right")
        ze = self.zk[top : top + int(self.hp["zext"] / self.hp["dz"])]
        W = np.empty((len(self.zk), 2))
        W[:top] = Vs[:top]
        W[top : top + len(ze)] = self._extrap(zs, Vs, ze)
        W[top + len(ze) :] = W[top + len(ze) - 1]
        C = np.vstack([np.zeros((1, 2)), np.cumsum(0.5 * (W[1:] + W[:-1]) * self.hp["dz"], 0)])
        return W, C

    # ---------------------------------------------------------------- prediction
    def predict(self, states, status, t, idx, horizons):
        self.update(states, status, t)
        return self.propagate(states[-1, idx], np.asarray(horizons, float))

    def propagate(self, s, horizons):
        """Positions (len(s), H, 3) of balloons with states s (n, 6) after each horizon (s)."""
        h = horizons[None, :]
        out = s[:, None, :3] + s[:, None, 3:] * h[..., None]  # constant velocity (no profile yet / not rising)
        if self.prof is None:
            return out
        C = self.prof[1]
        z0, vz = s[:, None, 2], s[:, None, 5]
        rising = np.abs(vz) > 0.2
        vzs = np.where(rising, vz, 1.0)
        for k in (0, 1):
            dx = (np.interp(z0 + vz * h, self.zk, C[:, k]) - np.interp(z0, self.zk, C[:, k])) / vzs
            out[..., k] = np.where(rising, s[:, None, k] + dx, out[..., k])
        return out

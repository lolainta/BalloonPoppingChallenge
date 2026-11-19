"""Learned history-profile predictor ("learned" family): Bayesian wind-profile regression with priors learned on TRAIN.

Structure identified on TRAIN seeds (from data only; no scenario files or env internals):
  * the wind depends on altitude only and is a natural cubic spline through knots every 20 m (AGL -20, 0, 20, ...):
    fitting that spline to all balloon samples of a seed leaves ~0.001 m/s rms (a 25 m knot grid or a shifted 20 m grid
    leaves 0.3-0.5 m/s).  Knot values are iid (no knot-to-knot or x-y correlation), uniform around a constant mean, with
    a spread that shrinks with altitude (std ~2.7 m/s at the ground, ~1 m/s at 300 m).
  * a balloon's horizontal velocity is that wind seen with its own small altitude lag, v_i(t) = w(z_i(t) - L_i)
    (L_i ~ 0.19 +- 0.05 m, i.e. a ~0.03 s drag lag, not explained by vz); balloons rise at a constant vz.
Model: knot values p (J x 2) and lags L_i are estimated jointly from ALL past samples of all balloons (the profile is
time-invariant, so every past sample counts), with Gaussian priors whose parameters (knot mean, knot std vs altitude,
lag mean/std) fit() learns from TRAIN.  Future xy = integral of the posterior wind along z_i - L_i + vz t.  Above the
highest balloon the posterior extrapolates: the C2 spline ties the partly seen top segment to the next one or two
knots; beyond, the wind relaxes to the learned prior mean.  (A ridge on the remaining residual displacement, with
features such as the distance flown above the top balloon, top-wind minus prior mean and horizon, did not beat this
posterior in 2-fold CV on TRAIN, so none is applied.)
Cost is O(new samples) per update: each balloon keeps its normal-equation blocks as a 2nd-order expansion in its lag.
"""
import numpy as np

HK = 20.0  # (m) knot spacing; knot j at elev + HK * (j - 1)


class Predictor:
    def __init__(self, given, J=34, dz=0.25):
        self.elev = given["environment"]["elevation"]
        self.nb = given["balloon"]["num"]
        self.J = J  # natural cubic spline through knot values p: second derivatives M = Mm @ p
        A = np.zeros((J, J)); A[0, 0] = A[-1, -1] = 1.0
        D = np.zeros((J, J))
        for i in range(1, J - 1):
            A[i, i - 1:i + 2] = [1.0, 4.0, 1.0]; D[i, i - 1:i + 2] = np.array([1.0, -2.0, 1.0]) * 6.0 / HK**2
        self.Mm = np.linalg.solve(A, D)
        self.dz, self.zg = dz, self.elev + np.arange(-10.0, HK * (J - 2), dz)  # grid for w and its integral
        self.BG = self.basis(self.zg)[0]
        # priors: defaults = what fit() learns on TRAIN seeds 20000-20031
        self.mu = np.array([-1.37, 1.23])   # knot mean (m/s)
        self.sig = (2.67, 311.0)              # knot std = a * exp(-z_agl / b)
        self.L0, self.sigL = 0.195, 0.048    # balloon lag mean / std (m)
        # tuned by 2-fold CV over TRAIN seeds (flat optimum)
        self.infl = 1.6                       # knot prior std inflation (knots are uniform: less shrinkage than a Gaussian)
        self.r = 0.003                        # sample noise (m/s)
        self.reset()

    def basis(self, z):
        """Natural-spline cardinal basis and its 1st/2nd altitude derivatives at z -> 3 x (len(z), J)."""
        s = (np.asarray(z, float) - self.elev + HK) / HK
        i = np.clip(np.floor(s).astype(int), 0, self.J - 2)
        u = s - i
        Mi, Mj, ar, h = self.Mm[i], self.Mm[i + 1], np.arange(len(u)), HK
        b0 = (h * h / 6 * ((1 - u) ** 3 - (1 - u)))[:, None] * Mi + (h * h / 6 * (u**3 - u))[:, None] * Mj
        b1 = (h / 6 * (1 - 3 * (1 - u) ** 2))[:, None] * Mi + (h / 6 * (3 * u**2 - 1))[:, None] * Mj
        b2 = (1 - u)[:, None] * Mi + u[:, None] * Mj
        b0[ar, i] += 1 - u; b0[ar, i + 1] += u
        b1[ar, i] -= 1 / h; b1[ar, i + 1] += 1 / h
        return b0, b1, b2

    # ------------------------------------------------------------------ learning the priors (TRAIN seeds)
    def fit(self, train_data):
        """Fit each TRAIN seed's whole record with vague priors, then set the priors to the knot / lag statistics."""
        keep = self.mu, self.sig, self.sigL, self.infl
        knots, lags = [], []
        for d in train_data:
            self.mu, self.sig, self.sigL, self.infl = np.zeros(2), (100.0, 1e9), 0.1, 1.0
            self.reset()
            S, St, t = d["states"], d["status"], d["t"]
            for k in range(len(t)):
                self.update(S[k], St[k], t[k])
            for _ in range(3):
                self.solve()
            ztop = S[..., 2][St == 1].max()
            knots.append(self.P[: int((ztop - self.elev + HK) // HK) - 1])  # knots with data above them
            lags.append((self.L0 + self.d)[(St == 1).sum(0) > 100])
        if not knots:
            self.mu, self.sig, self.sigL, self.infl = keep; self.reset(); return
        n = min(len(k) for k in knots)
        K = np.array([k[:n] for k in knots])
        self.mu = K.reshape(-1, 2).mean(0)
        sd = np.sqrt(((K - self.mu) ** 2).mean((0, 2)))[1:]  # knot AGL 0 .. : log-linear fit of the std
        c = np.polyfit(HK * np.arange(len(sd)), np.log(sd), 1)
        self.sig = (float(np.exp(c[1])), float(-1.0 / c[0]))
        L = np.concatenate(lags)
        self.L0, self.sigL, self.infl = float(L.mean()), float(L.std()), keep[3]
        self.reset()

    # ------------------------------------------------------------------ incremental estimation
    def reset(self):
        J, n = self.J, self.nb
        zk = self.elev + HK * (np.arange(J) - 1.0)
        s2 = (self.infl * self.sig[0] * np.exp(-(zk - self.elev) / self.sig[1])) ** 2
        self.A_pri, self.Y_pri = np.diag(self.r**2 / s2), self.mu[None] * (self.r**2 / s2)[:, None]
        # per balloon, samples expanded around lag L0:  b(z - L0 - d) = b0 - d b1 + d^2/2 b2
        self.A0, self.S01, self.S11, self.S02 = (np.zeros((n, J, J)) for _ in range(4))
        self.Y0, self.E1, self.E2 = (np.zeros((n, J, 2)) for _ in range(3))
        self.d = np.zeros(n)  # lag - L0
        self.P = np.linalg.solve(self.A_pri, self.Y_pri)
        self.t_last, self.key, self.Wc, self.vz_set = -np.inf, None, None, 6.1

    def update(self, state, status, t):
        """Ingest one observation (balloon_states (N, 6), balloon_status (N,) or (N, 1), time); refit knots and lags."""
        w = 1.0 if not np.isfinite(self.t_last) else float(np.clip((t - self.t_last) / 0.1, 0.0, 1.0))  # 10 Hz units
        self.t_last = t
        f = np.flatnonzero((np.ravel(status) == 1) & (state[:, 2] > self.elev + 1.5))
        if len(f) > 2:  # settled climb rate, for balloons still in their ~0.2 s release transient
            self.vz_set = float(np.median(state[f, 5]))
        if len(f):
            b0, b1, b2 = self.basis(state[f, 2] - self.L0)
            v = state[f, 3:5].astype(float) * w
            wb0 = w * b0[:, :, None]
            self.A0[f] += wb0 * b0[:, None]; self.S01[f] += wb0 * b1[:, None]; self.S02[f] += wb0 * b2[:, None]
            self.S11[f] += w * b1[:, :, None] * b1[:, None]
            self.Y0[f] += b0[:, :, None] * v[:, None]; self.E1[f] += b1[:, :, None] * v[:, None]
            self.E2[f] += b2[:, :, None] * v[:, None]
        self.solve()

    def solve(self):
        d, d2 = self.d, self.d**2
        T = np.tensordot(d, self.S01, 1); U = np.tensordot(d2, self.S02, 1)
        A = self.A_pri + self.A0.sum(0) - T - T.T + np.tensordot(d2, self.S11, 1) + 0.5 * (U + U.T)
        Y = self.Y_pri + self.Y0.sum(0) - np.tensordot(d, self.E1, 1) + 0.5 * np.tensordot(d2, self.E2, 1)
        self.P = P = np.linalg.solve(A, Y)
        # each balloon's lag: Newton step on its squared residual (2nd-order model) with the lag prior
        tr = lambda M: np.einsum("ijc,jc->i", M @ P, P)  # noqa: E731
        g1 = np.einsum("ijc,jc->i", self.E1, P) - tr(self.S01)
        gn = tr(self.S11)
        h2 = np.maximum(gn + tr(self.S02) - np.einsum("ijc,jc->i", self.E2, P), 0.5 * gn)
        self.d = np.clip(-g1 / (h2 + (self.r / self.sigL) ** 2), -self.L0, 0.4)
        self.Wc = None

    def _ingest(self, states, status, t):
        key = float(states[0, :, :3].sum())  # new episode (other pad layout) or time going back -> start over
        if t[-1] < self.t_last or key != self.key:
            self.reset(); self.key = key
        for k in np.flatnonzero(t > self.t_last):
            self.update(states[k], status[k], t[k])

    # ------------------------------------------------------------------ prediction
    def _prop(self, s, L, horizons):
        """Positions (n, H, 3) at the horizons of balloons with states s (n, 6) and lags L (n,)."""
        if self.Wc is None:  # altitude integral of the posterior wind
            W = self.BG @ self.P
            self.Wc = np.vstack([np.zeros((1, 2)), np.cumsum(0.5 * (W[1:] + W[:-1]) * self.dz, 0)])
        vz = np.where(s[:, 2] > self.elev + 1.5, np.maximum(s[:, 5], 0.5), self.vz_set)
        z0 = s[:, 2] - L
        z1 = z0[:, None] + vz[:, None] * horizons[None]
        out = np.empty((len(s), len(horizons), 3))
        for c in (0, 1):
            Wc = self.Wc[:, c]
            out[..., c] = s[:, None, c] + (np.interp(z1, self.zg, Wc) - np.interp(z0, self.zg, Wc)[:, None]) / vz[:, None]
        out[..., 2] = s[:, None, 2] + vz[:, None] * horizons[None]
        return out

    def predict(self, states, status, t, idx, horizons):
        self._ingest(states, status, t)
        return self._prop(states[-1, idx].astype(float), self.L0 + self.d[idx], np.asarray(horizons, float))

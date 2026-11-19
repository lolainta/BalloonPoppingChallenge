"""Physical balloon predictor (family "phys").

Model (every piece verified / fitted on TRAIN seeds only, see fit()):
  * horizontal: quadratic drag on the 3-D air-relative velocity,
        dv_h/dt = k |w(z) - v| (w(z) - v)_h,   |w - v| includes the balloon's rise speed (wind is horizontal),
    k [1/m] fitted on TRAIN (~5 /m -> relaxation time ~0.03 s, altitude lag 1/k ~ 0.2 m).
  * vertical: terminal ascent at the balloon's own current vz (constant over a flight to ~1e-3 m/s); balloons still
    in the release transient (< 1.5 m above the pad) get the median vz of the settled balloons.
  * wind w(z): static in time. Every past observation of every balloon is inverted through the drag law
    (w = v + u,  k sqrt(|u|^2 + vz^2) u = dv/dt, central differences) and binned over altitude. TRAIN shows the profile
    is a cubic spline through node values every 20 m of altitude (nodes at multiples of 20 m ASL); node deviations from a
    constant mean are independent, uniform with an amplitude that decays with altitude. So w = natural spline of the
    node values, estimated as a posterior mean (Gaussian prior N(mu, a(z)^2/3) per node, observations = bin means,
    then the uniform prior applied by importance sampling to the few partly determined nodes near the top).
    Below the highest altitude seen this reproduces the profile (~0.01 m/s); above it, the posterior uses the shape of
    the partially seen spline segment and reverts to the TRAIN mean for unseen nodes.
  * prediction: integrate the drag ODE per balloon (vectorised, exponential integrator, dt = 0.1 s).
Uses only observations up to now (states[:k+1] ...) and the given parameters.
"""
import numpy as np
from scipy.interpolate import CubicSpline


def _slip(a, vz, k):
    """Air-relative horizontal velocity u = w - v from the drag acceleration a: k sqrt(|u|^2 + vz^2) u = a."""
    A = np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12
    q = (A / k) ** 2
    s = np.sqrt(2.0 * q / (np.sqrt(vz ** 4 + 4.0 * q) + vz ** 2 + 1e-12))  # stable root of s^2 (s^2 + vz^2) = q
    return a * (s / A)


def _wind_samples(S, St, t, j, k, zmin):
    """Wind samples (z, w) from centre rows j (needs rows j-1 and j+1)."""
    ok = (St[j - 1] == 1) & (St[j] == 1) & (St[j + 1] == 1) & (S[j, :, 2] > zmin)
    a = (S[j + 1, :, 3:5] - S[j - 1, :, 3:5]) / (t[j + 1] - t[j - 1])[:, None, None]
    s = S[j]
    return s[..., 2][ok], s[..., 3:5][ok] + _slip(a[ok], s[..., 5:6][ok], k)


class Predictor:
    def __init__(self, given, k=5.35, mu=(-1.34, 1.235), a0=4.63, H=312.6, node_dz=20.0, dz=0.1, sig_obs=0.01,
                 dt=0.1, box=True, ns=2000, ztop=840.0):
        self.elev = given["environment"]["elevation"]
        self.k, self.mu, self.a0, self.H, self.dz, self.sig_obs, self.dt = k, np.array(mu), a0, H, dz, sig_obs, dt
        self.zn = np.arange(0.0, ztop + 1e-6, node_dz)                       # spline nodes (m ASL)
        self.zg = np.arange(self.elev, ztop, dz)                             # profile grid (m ASL)
        self.G = CubicSpline(self.zn, np.eye(len(self.zn)), bc_type="natural")(self.zg)  # grid <- node values
        self.zmin = self.elev + 1.5                                          # skip the release transient
        self.box, self.eps = box, np.random.default_rng(0).standard_normal((ns, 8, 2))  # fixed draws: deterministic
        self.n, self.last = 0, None

    # ------------------------------------------------------------------ TRAIN fit (observable quantities only)
    def fit(self, data):
        """k: drag constant that makes the inverted wind samples most spline-like (smallest residual of a free spline
        fit on the node grid); mu, a0, H: mean and amplitude a(z) = a0 exp(-(z - elev) / H) of the node deviations."""
        seeds = []
        for d in data:
            S, St, t = d["states"].astype(float), d["status"], d["t"]
            j = np.arange(1, len(t) - 1)
            seeds.append((S, St, t, j))

        def nodes(k, want_res=False):
            F, R = [], []
            for S, St, t, j in (seeds[:12] if want_res else seeds):
                z, w = _wind_samples(S, St, t, j, k, self.zmin)
                Gs = CubicSpline(self.zn, np.eye(len(self.zn)), bc_type="natural")(z)
                use = self.zn < z.max() + 3 * (self.zn[1] - self.zn[0])
                c, *_ = np.linalg.lstsq(Gs[:, use], w, rcond=None)
                R.append(np.mean((Gs[:, use] @ c - w)[z < z.max() - 40] ** 2))
                f = np.full((len(self.zn), 2), np.nan); f[use] = c
                f[self.zn > z.max() - 10] = np.nan; f[self.zn < self.elev] = np.nan  # keep well-determined nodes
                F.append(f)
            return np.array(F), float(np.mean(R))

        ks = np.array([3.0, 4.0, 5.0, 6.5, 8.0])
        res = np.array([nodes(k, True)[1] for k in ks])
        i = int(np.clip(np.argmin(res), 1, len(ks) - 2))                     # parabola through the best 3 (in log k)
        x, y = np.log(ks[i - 1:i + 2]), res[i - 1:i + 2]
        p = np.polyfit(x, y, 2)
        self.k = float(np.exp(-p[1] / (2 * p[0]))) if p[0] > 0 else float(ks[i])
        F, _ = nodes(self.k)
        self.mu = np.nanmean(F.reshape(-1, 2), 0)
        D = np.abs(F - self.mu)
        n = np.sum(~np.isnan(D), axis=(0, 2))
        ok = n >= 20
        ahat = np.nanmax(D[:, ok], axis=(0, 2)) * (n[ok] + 1) / n[ok]      # uniform: E[max of n] = a n / (n + 1)
        b = np.polyfit(self.zn[ok] - self.elev, np.log(ahat), 1)
        self.a0, self.H = float(np.exp(b[1])), float(-1.0 / b[0])
        self.n, self.last = 0, None
        return self

    # ------------------------------------------------------------------ wind profile from all past observations
    def _reset(self):
        self.sw = np.zeros((len(self.zg), 2)); self.cnt = np.zeros(len(self.zg)); self.n = 0
        self.W = np.tile(self.mu, (len(self.zg), 1))

    def update(self, states, status, t):
        """Add the new rows' wind samples and re-solve the node posterior (~1-5 ms; call every few steps)."""
        if self.last is None or len(t) < self.n or not np.array_equal(states[self.n - 1], self.last):
            self._reset()                                                    # new episode
        elif len(t) == self.n:
            return
        j = np.arange(max(self.n - 1, 1), len(t) - 1)                        # new centre rows (central differences)
        self.n, self.last = len(t), states[-1].copy()
        if len(j):
            z, w = _wind_samples(states, status, t, j, self.k, self.zmin)
            i = np.clip(np.round((z - self.zg[0]) / self.dz).astype(int), 0, len(self.zg) - 1)
            np.add.at(self.cnt, i, 1.0); np.add.at(self.sw, i, w)
        cnt, sw = self.cnt, self.sw
        if len(t) >= 3:                                                      # newest row: 2nd-order backward difference
            S, q = states, len(t) - 1                                        # (used now, replaced next update)
            ok = (status[q] == 1) & (status[q - 1] == 1) & (status[q - 2] == 1) & (S[q, :, 2] > self.zmin)
            a = (3 * S[q, ok, 3:5] - 4 * S[q - 1, ok, 3:5] + S[q - 2, ok, 3:5]) / (t[q] - t[q - 2])
            i = np.clip(np.round((S[q, ok, 2] - self.zg[0]) / self.dz).astype(int), 0, len(self.zg) - 1)
            cnt, sw = cnt.copy(), sw.copy()
            np.add.at(cnt, i, 1.0); np.add.at(sw, i, S[q, ok, 3:5] + _slip(a, S[q, ok, 5:6], self.k))
        ok = cnt > 0
        if not ok.any():
            return
        Go = self.G[ok]
        amp = self.a0 * np.exp(-(self.zn - self.elev) / self.H)              # node deviation ~ U(-amp, amp)
        prior = 3.0 / amp ** 2                                               # Gaussian with the same variance
        A = Go.T @ Go / self.sig_obs ** 2 + np.diag(prior)
        f = np.linalg.solve(A, Go.T @ (sw[ok] / cnt[ok, None] - self.mu) / self.sig_obs ** 2)
        if self.box:
            f = self._box(f, np.linalg.inv(A), amp, prior)
        self.W = self.G @ f + self.mu

    def _box(self, f, P, amp, prior):
        """Posterior mean under the true (uniform) node prior for the partly determined nodes: importance-reweight
        draws of the Gaussian posterior by U(-amp, amp) / N(0, amp^2/3), then shift the other nodes accordingly."""
        sd = np.sqrt(np.diag(P))
        R = np.flatnonzero((sd > 0.03 * amp) & (sd < 0.97 * amp / np.sqrt(3)))[:self.eps.shape[1]]
        if not len(R):
            return f
        PR = P[np.ix_(R, R)]
        fs = f[R] + np.einsum("ij,sjc->sic", np.linalg.cholesky(PR), self.eps[:, :len(R)])  # (ns, R, 2)
        lw = np.where(np.abs(fs) <= amp[R, None], 0.5 * fs ** 2 * prior[R, None], -np.inf).sum(1)
        if not np.isfinite(lw.max(0)).all():
            return f
        wt = np.exp(lw - lw.max(0)); wt /= wt.sum(0)
        fR = np.einsum("sc,sic->ic", wt, fs)
        f = f + P[:, R] @ np.linalg.solve(PR, fR - f[R])                    # conditional shift (exact on R)
        return f

    # ------------------------------------------------------------------ prediction
    def predict(self, states, status, t, idx, horizons, update=True):
        """(len(idx), len(horizons), 3) positions. update=False reuses the last profile (agent: update() every 5 steps)."""
        if update:
            self.update(states, status, t)
        s = states[-1, idx].astype(float)
        p, v, vz = s[:, :3].copy(), s[:, 3:5].copy(), s[:, 5].copy()
        cur = states[-1]
        settled = (status[-1] == 1) & (cur[:, 2] > self.elev + 3.0)
        young = s[:, 2] < self.zmin                                          # release transient: vz overshoots
        vz[young] = np.median(cur[settled, 5]) if settled.any() else 6.0
        dt = self.dt
        steps = np.round(np.asarray(horizons) / dt).astype(int)
        tm = (np.arange(steps.max()) + 0.5) * dt
        zm = p[:, 2:3] + vz[:, None] * tm[None]                              # altitude at mid-step
        wm = np.stack([np.interp(zm, self.zg, self.W[:, c]) for c in (0, 1)], -1)  # (n, T, 2)
        XY = np.empty((len(idx), steps.max() + 1, 2)); XY[:, 0] = p[:, :2]
        vz2 = vz[:, None] ** 2
        for n in range(steps.max()):                                         # exact for w constant over the step
            w = wm[:, n]
            du = v - w
            c = np.maximum(self.k * np.sqrt(np.sum(du * du, -1, keepdims=True) + vz2), 1e-6)
            e = np.exp(-c * dt)
            XY[:, n + 1] = XY[:, n] + w * dt + du * (1.0 - e) / c
            v = w + du * e
        return np.concatenate([XY[:, steps], (p[:, 2:3] + vz[:, None] * steps * dt)[..., None]], -1)

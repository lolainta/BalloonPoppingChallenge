"""How far ahead can a balloon's position be predicted from the observation alone? (DECISION_LOG §206)
The scenario-4 gust is a static altitude profile (a node every 20 m, drawn at reset), so every airborne balloon is a
wind sensor: a balloon now at height h shows the horizontal velocity another balloon will have when it rises to h.
Compares, for airborne balloons at t0, the true position at t0 + dt with
  cv:   p + v dt                                      (what the agent's reach-time features assume)
  wind: z(tau) = z + vz tau, horizontal velocity at tau = that of the balloons now at z(tau) (interpolated by height;
        above the highest balloon: the highest's)    (observation only: current balloon_states / status)
  gnc:  the agent's own predictor (gnc_agent bwind=4: history profile, fed every 0.1 s from t = 0 like get_action)
   .venv/bin/python dev/wind_probe.py SEED [SEED ...]   (seeds recorded in the seed registry)"""
import contextlib, io, sys

import numpy as np

sys.path.insert(0, ".")
from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters
from dev.seed_registry import record

T0S, DTS = (48.0, 55.0, 62.0), (2.0, 5.0, 10.0, 20.0)
seeds = [int(s) for s in sys.argv[1:]]
record(seeds, "wind_probe", note="balloon position prediction cv vs wind profile (§206)", scenario=4)
params, given = load_scenario_parameters(4)
err = {(m, dt): [] for m in ("cv", "wind", "gnc") for dt in DTS}
for seed in seeds:
    env = BalloonPoppingEnv(render_mode=None, parameters=params)
    with contextlib.redirect_stdout(io.StringIO()):
        env.reset(seed=seed)
    u = env.unwrapped
    F, rel, h = u._balloon_flights, np.asarray(u._balloon_release_at_step), u.simulation_parameters["time_step"]
    from BalloonPoppingGymEnv.agents.gnc_agent import GNCAgent
    ag, fed = GNCAgent(given, bwind=4), 0
    feed = lambda s: ag._hist_add({"balloon_states": F[:, :, s], "balloon_status": (rel <= s).astype(int)[:, None],
                                   "simulation_time": s * h})
    for t0 in T0S:
        k = int(round(t0 / h))
        for s in range(fed, k + 1, 10):
            feed(s)
        fed = (k // 10 + 1) * 10
        S = F[:, :, k]
        air = (rel <= k - int(1.0 / h))  # released at least 1 s ago (off the ground)
        z_all, vh_all = S[air, 2], S[air, 3:5]
        o = np.argsort(z_all); zs, vs = z_all[o], vh_all[o]
        wind = lambda z: np.stack([np.interp(z, zs, vs[:, 0]), np.interp(z, zs, vs[:, 1])], -1)
        for i in np.flatnonzero(air):
            for dt in DTS:
                k1 = k + int(round(dt / h))
                if k1 >= F.shape[2]:
                    continue
                true = F[i, :2, k1]
                err["cv", dt].append(np.linalg.norm(S[i, :2] + S[i, 3:5] * dt - true))
                tau = np.arange(0.0, dt, 0.1) + 0.05
                p = S[i, :2] + (wind(S[i, 2] + S[i, 5] * tau) * 0.1).sum(0)
                err["wind", dt].append(np.linalg.norm(p - true))
                g = S[i, :2] + S[i, 3:5] * dt + ag._bdrift(S[i, 2], S[i, 5], dt, ag.hprof)[:2]
                err["gnc", dt].append(np.linalg.norm(g - true))
    print(f"seed {seed} done", flush=True)
print("horizontal position error (m): median / p90 over airborne balloons at t0 =", T0S)
for dt in DTS:
    c, w, g = np.array(err["cv", dt]), np.array(err["wind", dt]), np.array(err["gnc", dt])
    print(f"  dt {dt:4.0f} s  n {len(c):5d}   cv {np.median(c):6.2f} / {np.percentile(c, 90):6.2f}   "
          f"wind {np.median(w):6.2f} / {np.percentile(w, 90):6.2f}   gnc {np.median(g):6.2f} / {np.percentile(g, 90):6.2f}")

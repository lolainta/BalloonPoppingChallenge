"""Scratch prototype (not in the repo): GNSS-aided attitude correction for scenario 40's noisy gyro (DECISION_LOG §236).
The navigation filter's velocity innovation (v_gnss - v_hat) settles at ~ (delta x f_w) dt / k_vel when the attitude
estimate is tilted by delta (f_w = the predicted world specific force) -> delta_est = k_vel / dt * (f_w x innov) / |f_w|^2,
rotate q by g * delta_est each step (Mahony-style; the roll about f is unobservable and left alone).
   cd BalloonPoppingChallenge && AF_G=0.02 .venv/bin/python <this> SEED_LO SEED_HI SCENARIO [kw ...]"""
import multiprocessing as mp, os, sys

import numpy as np

sys.path.insert(0, ".")
import dev.run_seeds as R
from BalloonPoppingGymEnv.agents.gnc_agent import quat_step, quat_to_R
from BalloonPoppingGymEnv.agents.rl3_agent import RL3Agent

G_AF = float(os.environ.get("AF_G", 0.02))


class AttFix(RL3Agent):
    def _navigate(self, acc, p_gnss, v_gnss):
        if self.p_hat is not None and G_AF > 0:
            Rq = quat_to_R(self.q); f = Rq @ acc; ff = f @ f
            if ff > 9.0:  # thrusting (|f| > 3 m/s^2): the tilt is observable
                d = self.k_vel / self.dt * np.cross(f, v_gnss - self.v_hat) / ff  # world-frame tilt estimate (rad)
                self.q = quat_step(self.q, Rq.T @ (G_AF * d) / self.dt, self.dt)  # rotate by g * d (body frame)
        return super()._navigate(acc, p_gnss, v_gnss)


import BalloonPoppingGymEnv.agents.rl3_agent as _M
_M.RL3Agent = AttFix  # run_seeds imports RL3Agent inside run(): replace it in its module
if __name__ == "__main__":
    lo, hi, sc = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    kw = R.parse_kw(sys.argv[4:])
    from dev.seed_registry import record
    record(range(lo, hi + 1), "attfix_probe", f"AF_G={G_AF} {' '.join(sys.argv[4:])}", sc)
    os.environ["AGENT"] = "rl3"
    with mp.Pool(min(hi - lo + 1, len(os.sched_getaffinity(0)))) as pool:
        res = pool.map(R.run, [(s, sc, dict(kw)) for s in range(lo, hi + 1)])
    print(f"AF_G={G_AF}  mean pops {np.mean([r[1] for r in res]):.3f}  mean max_att_err {np.mean([r[3] for r in res]):.1f} deg  "
          f"nav_vel {np.mean([r[9][2] for r in res]):.3f}  (n={len(res)})", flush=True)

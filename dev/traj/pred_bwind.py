"""Current GNC predictor (bwind): wind profile from the CURRENT snapshot (Gaussian kernel 5 m over altitude), balloons
follow it as they rise at their current vertical speed (GNCAgent._wind_profile / _bdrift)."""
import numpy as np
from BalloonPoppingGymEnv.agents.gnc_agent import GNCAgent


class Predictor:
    def __init__(self, given):
        self.g = GNCAgent(given, bwind=1)

    def predict(self, states, status, t, idx, horizons):
        obs = {"balloon_states": states[-1], "balloon_status": status[-1][:, None]}
        self.g.wprof = self.g._wind_profile(obs)
        s = states[-1, idx]
        cv = s[:, None, :3] + s[:, None, 3:] * horizons[None, :, None]
        return cv + self.g._bdrift(s[:, None, 2], s[:, None, 5], horizons[None, :])

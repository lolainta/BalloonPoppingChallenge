"""GNCAgent bwind=4 (history profile inside the agent, constant above the highest balloon) through the harness."""
import numpy as np
from BalloonPoppingGymEnv.agents.gnc_agent import GNCAgent


class Predictor:
    def __init__(self, given):
        self.given, self.g, self.first, self.n = given, None, None, 0

    def predict(self, states, status, t, idx, horizons):
        if self.g is None or not np.array_equal(states[0], self.first):  # new seed
            self.g, self.first, self.n = GNCAgent(self.given, bwind=4), states[0].copy(), 0
        for k in range(self.n, len(t)):  # rows are 0.1 s apart = the agent's every-10-steps update
            self.g._hist_add({"balloon_states": states[k], "balloon_status": status[k][:, None], "simulation_time": t[k]})
        self.n = len(t)
        s = states[-1, idx]
        return s[:, None, :3] + s[:, None, 3:] * horizons[None, :, None] + self.g._bdrift(s[:, None, 2], s[:, None, 5], horizons[None, :])

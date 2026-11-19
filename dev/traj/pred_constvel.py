"""Baseline: constant velocity."""
import numpy as np


class Predictor:
    def __init__(self, given):
        pass

    def predict(self, states, status, t, idx, horizons):
        s = states[-1, idx]
        return s[:, None, :3] + s[:, None, 3:] * horizons[None, :, None]

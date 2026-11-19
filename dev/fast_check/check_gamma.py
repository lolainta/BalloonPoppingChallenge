"""B (discount): async_ppo.gae / gammas / to_demo-style returns on synthetic sequences, RL3_FAST set. The same reward flown as 0.5 s FULL
rows or split into 0.1 s FAST rows gets the same discount (per 0.5 s: GAMMA, GAE weight GAMMA LAM); pad rows (RL3_WAIT_NODISCOUNT)
discount 1; a pop-shortened row (dt 0.2 s) discounts GAMMA ** 0.4; nothing crosses a done.
source dev/fast_check/env.sh; RL3_FAST=10 RL3_WAIT_NODISCOUNT=1 $PY dev/fast_check/check_gamma.py"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dev import async_ppo as ap  # noqa: E402

assert Path(ap.__file__).resolve().is_relative_to(ROOT) and ap.FAST and ap.WAIT_NODISCOUNT


def seg(dt, rew, pad=0, done=None):
    n = len(dt)
    obs = np.zeros((n, 64 + 1500), np.float32); obs[pad:, ap.LAUNCHED_COL] = 1.0
    return {"obs": obs, "rew": np.array(rew, float), "val": np.zeros(n), "done": np.zeros(n, bool) if done is None else np.array(done),
            "dt": np.array(dt, float)}


def ret(s):  # discounted return-to-go (= learn.to_demo)
    R, g, gam = np.zeros(len(s["rew"])), 0.0, ap.gammas(s["obs"], s["dt"])
    for t in reversed(range(len(R))):
        g = s["rew"][t] + gam[t] * g; R[t] = g
    return R


G, L = ap.GAMMA, ap.LAM
# a pop reward 1.0 s after the first row: 2 FULL rows of 0.5 s + the row that pops, or 10 FAST rows of 0.1 s + that row
full = seg([0.5, 0.5, 0.5], [0, 0, 1])
fast = seg([0.1] * 10 + [0.5], [0] * 10 + [1])
a_full, a_fast = ap.gae(full, 0.0), ap.gae(fast, 0.0)
print(f"GAE weight of a reward 1.0 s ahead (zero values): 0.5 s rows {a_full[0]:.6f}, 0.1 s rows {a_fast[0]:.6f}, (GAMMA LAM)^2 = {(G * L) ** 2:.6f}")
print(f"return-to-go of it: 0.5 s rows {ret(full)[0]:.6f}, 0.1 s rows {ret(fast)[0]:.6f}, GAMMA^2 = {G ** 2:.6f}; at 0.5 s: {ret(fast)[5]:.6f} "
      f"(GAMMA {G})")
print(f"per-row gamma: 0.1 s {ap.gammas(fast['obs'], fast['dt'])[0]:.6f} (GAMMA^0.2 {G ** 0.2:.6f}), pop-shortened 0.2 s "
      f"{ap.gammas(fast['obs'][:1], [0.2])[0]:.6f} (GAMMA^0.4 {G ** 0.4:.6f}), lambda 0.1 s {ap.lams([0.1])[0]:.6f} (LAM^0.2 {L ** 0.2:.6f})")
p = seg([0.5] * 3 + [0.1] * 3, [0] * 5 + [1], pad=3)
print(f"pad rows (3, WAIT_NODISCOUNT): gamma {ap.gammas(p['obs'], p['dt'])[:3]}, return-to-go {np.round(ret(p), 6)} (GAMMA^0.4 = "
      f"{G ** 0.4:.6f} at the launch row)")
d = seg([0.1] * 4, [0, 1, 0, 5], done=[0, 1, 0, 0])
print(f"done after row 1: GAE {np.round(ap.gae(d, 0.0), 6)} (row 0 = GAMMA^0.2 LAM^0.2 = {(G * L) ** 0.2:.6f}; rows 2-3 not mixed in)")
v = seg([0.1] * 5, [0] * 5); v["val"] = np.ones(5)
print(f"constant values 1, no reward, boot 1: GAE {np.round(ap.gae(v, 1.0), 6)} (per row GAMMA^0.2 - 1 = {G ** 0.2 - 1:.6f} + ...)")

"""Gym wrapper for RL guidance training: one step = one RLAgent decision (see agents/rl_agent.py).
Reward = balloons popped until the next decision; after burnout the flight coasts to the end and the
remaining pops go to the last step."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import contextlib, io, logging, sys, threading, warnings
from pathlib import Path

import gymnasium as gym
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from BalloonPoppingGymEnv.agents.rl_agent import K, N_OBS, STRETCH, RLAgent  # noqa: E402
from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv  # noqa: E402
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters  # noqa: E402


class GuidanceEnv(gym.Env):
    def __init__(self, seed_lo=100_000, seed_hi=10_000_000, agent_kw=None):
        warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
        self.params, self.given = load_scenario_parameters(1)
        self.action_space = gym.spaces.MultiDiscrete([K, len(STRETCH)])
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (N_OBS,), np.float32)
        self.seed_lo, self.seed_hi, self.agent_kw = seed_lo, seed_hi, dict(agent_kw or {})
        self.env = None
        self._next = None  # (thread, result box) preparing the next episode

    def _burning(self):
        return self.ag.p_hat is not None and self.ag._burning(float(self.obs["simulation_time"]))

    def _prepare(self, seeds, box):
        """Build an episode up to the first decision (env reset + pre-launch flight). Runs in a thread,
        using the time this worker would otherwise spend waiting for the slowest env."""
        for s in seeds:  # some env seeds crash in reset (negative balloon mass): try the next
            try:
                env = BalloonPoppingEnv(render_mode=None, parameters=self.params)
                with contextlib.redirect_stdout(io.StringIO()):
                    obs, _ = env.reset(seed=s)
                break
            except Exception:
                continue
        ag = RLAgent(self.given, use_policy=0, **self.agent_kw)
        done = False
        while not (done or (ag.p_hat is not None and ag._burning(float(obs["simulation_time"])))):
            obs, _, te, tr, _ = env.step(ag.get_action(obs))
            done = te or tr
        box.append((env, obs, ag, s))

    def _spawn(self):
        seeds = [int(x) for x in self.np_random.integers(self.seed_lo, self.seed_hi, 10)]
        box = []
        th = threading.Thread(target=self._prepare, args=(seeds, box), daemon=True)
        th.start()
        self._next = (th, box)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self._next is None or seed is not None:
            self._spawn()
        th, box = self._next
        th.join()
        self.env, self.obs, self.ag, s = box[0]
        self.done = False
        self.seed_cur, self.last_pop = s, None
        if os.environ.get("PREFETCH", "1") == "1":
            self._spawn()  # prefetch the next episode while this one runs
        else:
            self._next = None
        return self.ag.features(self.obs), {"seed": s}

    def _step_env(self):
        self.obs, r, te, tr, self.info = self.env.step(self.ag.get_action(self.obs))
        self.done = te or tr
        if r > 0:
            self.last_pop = float(self.obs["simulation_time"])
        return r

    def step(self, action):
        self.ag.decide(self.obs, action)
        reward = 0.0
        while not self.done:
            reward += self._step_env()
            if not self._burning():  # burnout: coast to the end, no more decisions
                while not self.done:
                    reward += self._step_env()
                break
            if self.ag.needs_decision(self.obs):
                break
        info = {"seed": self.seed_cur, "last_pop": self.last_pop} if self.done else {}
        return self.ag.features(self.obs), reward, self.done, False, info


if __name__ == "__main__":  # smoke test: GNC-like fixed policy (nearest candidate, stretch 1.3)
    import time
    e = GuidanceEnv(); t0 = time.time()
    x, info = e.reset(seed=0); tot, n = 0.0, 0
    while True:
        x, r, d, _, _ = e.step(np.array([0, 3])); tot += r; n += 1
        if d: break
    print(f"seed {info['seed']}: decisions {n}  pops {tot}  wall {time.time()-t0:.0f}s  obs {x.shape}")

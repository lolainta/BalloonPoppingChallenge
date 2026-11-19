"""RL3 training env: RL2 env + pre-launch decisions (launch now / wait), see agents/rl3_agent.py."""
import os

import gymnasium as gym
import numpy as np

from BalloonPoppingGymEnv.agents.rl2_agent import N_MAX, STRETCH
from BalloonPoppingGymEnv.agents.rl3_agent import ATT, DECIDE, N_OBS, T_MAX, T_MIN, RL3Agent
from dev import rl2_env


def dr_params(params, spec, rng=None):
    """Domain randomization of hidden scenario parameters, one draw per balloon world (qualification params "might be
    different", #168). spec: 'k1,k2=lo:hi;k3=lo:hi' -> uniform draw, keys joined by ',' share one draw."""
    import copy
    from dev.run_seeds import set_param
    rng = rng or np.random.default_rng()
    p = copy.deepcopy(params)
    for entry in filter(None, spec.split(";")):
        keys, rng_s = entry.split("=")
        lo, hi = map(float, rng_s.split(":"))
        v = float(rng.uniform(lo, hi))
        for k in keys.split(","):
            set_param(p, k, v)
    return p


class GuidanceEnv3(rl2_env.GuidanceEnv):
    def __init__(self, **kw):
        super().__init__(**kw)
        if os.environ.get("RL3_T_MIN"):  # earliest launch decision used for training (default T_MIN)
            self.agent_kw.setdefault("t_min", float(os.environ["RL3_T_MIN"]))
        self.action_space = gym.spaces.MultiDiscrete([N_MAX, len(STRETCH), 2] + ([len(ATT)] if os.environ.get("RL3_ATT") else []))
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (N_OBS,), np.float32)

    def _prepare(self, seeds, box):
        import copy
        # RL3_REUSE=K: fly each balloon world K times (deep copies of the freshly reset env, identical to a new reset):
        # the scenario-4 reset (balloon Monte Carlo in the gust, ~4x the integration steps) costs 2-3 episodes
        reuse = int(os.environ.get("RL3_REUSE", 1))
        w = getattr(self, "_world", None)
        if w is None or w[2] >= reuse:
            params = dr_params(self.params, os.environ["RL3_DR"]) if os.environ.get("RL3_DR") else self.params
            for s in seeds:
                try:
                    env = rl2_env.BalloonPoppingEnv(render_mode=None, parameters=params)
                    obs, _ = rl2_env.reset_clean(env, s)
                    break
                except Exception:
                    continue
            w = self._world = [env, obs, 0, s]
        w[2] += 1
        env, obs, s = (copy.deepcopy(w[0]), copy.deepcopy(w[1]), w[3]) if w[2] < reuse else (w[0], w[1], w[3])
        kw = dict(self.agent_kw)
        if os.environ.get("RL3_FLOOR"):  # "lo:hi": a random launch floor per episode (exploring starts), so launching
            lo, hi = map(float, os.environ["RL3_FLOOR"].split(":"))  # at every time gets flown and valued
            kw["t_min"] = float(np.random.default_rng().uniform(lo, hi))
        ag = RL3Agent(self.given, use_policy=0, **kw)
        while float(obs["simulation_time"]) < ag.t_min:  # nothing to decide before the launch floor
            obs, *_ = env.step(ag.get_action(obs))
        box.append((env, obs, ag, s))

    def step(self, action):
        if self.ag.launched():
            # learned guidance residual: the bound of the policy that acts NOW (a worker reload updates agent_kw), also in an
            # episode built before that reload, so the env applies u exactly when the policy samples it; u = the last 3 columns
            rb = float(self.agent_kw.get("res_bound", 0.0))
            if rb or self.ag.res.any():
                if rb and len(action) < 6:
                    raise ValueError(f"res_bound expects [i, s, launch, .., u1, u2, u3], got {action!r}")
                self.ag.res_bound = rb
                self.ag.set_res(action[-3:] if rb else np.zeros(3))
            return super().step(action)
        if os.environ.get("RL3_ATTC"):  # continuous launch attitude: action = [0, 3, launch, u1, u2]
            if len(action) < 5:
                raise ValueError(f"RL3_ATTC expects [0, 3, launch, u1, u2], got {action!r} (policy without an attc head?)")
            att = np.asarray(action[3:5], float)
        else:
            att = int(action[3]) if len(action) > 3 else 0
        self.ag.decide_launch(self.obs, int(action[2]) == 1, att)
        reward = 0.0
        while not self.done:
            reward += self._step_env()
            if self.ag.launched():
                if self._burning():  # first post-launch decision
                    break
            elif self.ag.needs_launch_decision(self.obs) or float(self.obs["simulation_time"]) >= T_MAX:
                break
        self._x_last = self.ag.features(self.obs)
        info = {"seed": self.seed_cur, "last_pop": self.last_pop} if self.done else {}
        return self._x_last, reward, self.done, False, info

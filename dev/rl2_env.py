"""Gym wrapper for RL2 guidance training (agents/rl2_agent.py): one step = one decision.
Also logs every decision's outcome for the learned reach model (T1DATA dir) and hot-reloads that model (SLMODEL).
Reward = balloons popped until the next decision; after burnout the flight coasts to the end and the
remaining pops go to the last step."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import contextlib, io, logging, shutil, sys, tempfile, threading, time, warnings
from pathlib import Path

import gymnasium as gym
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from BalloonPoppingGymEnv.agents.rl2_agent import F, F_BASE, G_DIM, N_MAX, N_OBS, STRETCH, RL2Agent as RLAgent, parse_gnc_kw  # noqa: E402
from BalloonPoppingGymEnv.agents.rl2_sl import load_sl  # noqa: E402
from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv  # noqa: E402
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters  # noqa: E402


def reset_clean(env, seed):
    """env.reset inside a per-process temp dir that is emptied afterwards. The balloon Monte Carlo keeps its
    files (~160 MB) when a seed crashes or the worker is killed mid-reset; that filled the nodes' RAM-backed /tmp."""
    d = Path("/tmp") / f"bpc_{os.getpid()}"
    d.mkdir(exist_ok=True); tempfile.tempdir = str(d)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return env.reset(seed=seed)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def clean_stale_tmp():
    """Remove temp dirs left by dead workers (bpc_<pid>) and old leftovers from before reset_clean."""
    for d in Path("/tmp").glob("bpc_*"):
        try:
            if d.owner() == os.environ.get("USER") and not Path(f"/proc/{d.name[4:]}").exists():
                shutil.rmtree(d, ignore_errors=True)
        except (OSError, KeyError):
            pass
    for d in Path("/tmp").glob("balloon_sim_*"):
        try:
            if d.owner() == os.environ.get("USER") and time.time() - d.stat().st_mtime > 1800:
                shutil.rmtree(d, ignore_errors=True)
        except (OSError, KeyError):
            pass


if os.environ.get("RL3_FASTWIND"):  # training only (evaluation uses the official env): wind + gust tabulated on a
    from rocketpy import Environment, Function  # 0.5 m grid (~1e-3 m/s off the spline) -> one linear lookup instead
    _add_gust = Environment.add_wind_gust  # of nested Function sums: scenario-4 episodes ~2x faster

    def _fast_gust(self, gx, gy):
        _add_gust(self, gx, gy)
        z = np.arange(0.0, self.elevation + 3000.0, 0.5)  # above 3 km (never reached) held constant
        for ax in ("x", "y"):
            f = getattr(self, f"wind_velocity_{ax}")
            tab = np.column_stack([z, [f.get_value_opt(h) for h in z]])
            getattr(self, f"_Environment__set_wind_velocity_{ax}_function")(Function(tab, extrapolation="constant"))

    Environment.add_wind_gust = _fast_gust


class GuidanceEnv(gym.Env):
    def __init__(self, seed_lo=100_000, seed_hi=10_000_000, agent_kw=None):
        warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
        self.params, self.given = load_scenario_parameters(int(os.environ.get("RL3_SCEN", 1)))  # qualification: 4
        self.action_space = gym.spaces.MultiDiscrete([N_MAX, len(STRETCH)])
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (N_OBS,), np.float32)
        self.seed_lo, self.seed_hi, self.agent_kw = seed_lo, seed_hi, dict(agent_kw or {})
        for k, v in parse_gnc_kw(os.environ.get("GNC_KW")).items():  # flight tunables of this run (see rl3_net.export)
            self.agent_kw.setdefault(k, v)
        self.env = None
        self._next = None  # (thread, result box) preparing the next episode
        self.log_dir = os.environ.get("T1DATA")
        self.sl_path, self.sl_mtime, self.sl = os.environ.get("SLMODEL"), None, None
        self.rows, self.pending, self.n_eps = [], [], 0

    def _burning(self):
        return self.ag.p_hat is not None and self.ag._burning(float(self.obs["simulation_time"]))

    def _prepare(self, seeds, box):
        """Build an episode up to the first decision (env reset + pre-launch flight). Runs in a thread,
        using the time this worker would otherwise spend waiting for the slowest env."""
        for s in seeds:  # some env seeds crash in reset (negative balloon mass): try the next
            try:
                env = BalloonPoppingEnv(render_mode=None, parameters=self.params)
                obs, _ = reset_clean(env, s)
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
        if os.environ.get("SEED_LIST"):  # fine-tuning on a fixed set of seeds (leaderboard only); "@file" too
            sl = os.environ["SEED_LIST"]
            pool = [int(x) for x in (open(sl[1:]).read() if sl.startswith("@") else sl).replace(",", " ").split()]
            seeds = [int(x) for x in self.np_random.choice(pool, 10)]
        else:
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
        self.done = self.fast = False
        self.seed_cur, self.last_pop = s, None
        if self.sl_path and os.path.exists(self.sl_path) and os.path.getmtime(self.sl_path) != self.sl_mtime:
            self.sl_mtime = os.path.getmtime(self.sl_path)
            self.sl = load_sl(self.sl_path)
        self.ag.sl = self.sl
        self._flush_episode()
        if os.environ.get("PREFETCH", "1") == "1":
            self._spawn()  # prefetch the next episode while this one runs
        else:
            self._next = None
        self._x_last = self.ag.features(self.obs)
        return self._x_last, {"seed": s}

    def _step_env(self):
        self.obs, r, te, tr, self.info = self.env.step(self.ag.get_action(self.obs))
        self.done = te or tr
        if r > 0:
            self.last_pop = float(self.obs["simulation_time"])
        return r

    # ---------------------------------------------------------------- reach-model data
    def _log_decision(self, x, action):
        if not self.log_dir or self.ag.cmd is None or getattr(self.ag, "plan_on", False):  # planner flies: not the target's outcome
            return
        g = len(x) - N_MAX * F  # RL2: 62 global features, RL3: 64
        tok = x[g:].reshape(N_MAX, F)[self.ag.cmd[0], :F_BASE].copy()  # (the SL's 15 columns, §288)
        self.pending.append((np.concatenate([x[:6], tok, [action[1]]]), self.ag.cmd[0], float(self.obs["simulation_time"]), None))

    def _resolve(self, final=False):
        """Close pending samples whose target was popped (hit) or abandoned / out of time (miss).
        2026-10-14 (DECISION_LOG §189): the policy hands over to its next target ~0.5 s before contact and the old one
        pops on momentum, so a target that stops being the current one stays open for HOLD_S s (was: a miss at once,
        which labelled most hits as misses)."""
        HOLD_S = 1.0
        st = self.obs["balloon_status"][:, 0]
        t = float(self.obs["simulation_time"])
        cur = self.ag.cmd[0] if (self.ag.cmd is not None and not final) else -1
        keep = []
        for feat, b, t0, left in self.pending:
            if st[b] == 2:
                self.rows.append(np.concatenate([feat, [1.0, t - t0]]))
            elif st[b] != 1 or final:
                self.rows.append(np.concatenate([feat, [0.0, (left if left is not None else t) - t0]]))
            elif b != cur:
                left = t if left is None else left
                if t - left > HOLD_S:
                    self.rows.append(np.concatenate([feat, [0.0, left - t0]]))
                else:
                    keep.append((feat, b, t0, left))
            else:
                keep.append((feat, b, t0, None))
        self.pending = keep

    def _flush_episode(self):
        self.n_eps += 1
        if self.log_dir and self.rows and self.n_eps % 5 == 0:
            os.makedirs(self.log_dir, exist_ok=True)
            np.save(os.path.join(self.log_dir, f"{os.getpid()}_{time.time_ns()}.npy"), np.array(self.rows, np.float32))
            self.rows = []

    def step(self, action, fast=False):
        """fast (RL3_FAST, dev/rl3_env.py): a FAST step, target / gear held (no decide). With agent_kw fast_k = k the flight also
        stops every k sim steps since the decision (self.fast = True: the next step is a FAST one; = rl3_agent.fast_tick)."""
        x_prev = self._x_last if hasattr(self, "_x_last") else None
        if not fast:
            self.ag.decide(self.obs, action)
            if x_prev is not None:
                self._log_decision(x_prev, action)
        k, self.fast = int(self.agent_kw.get("fast_k", 0)), False
        reward = 0.0
        while not self.done:
            reward += self._step_env()
            if not self._burning():  # burnout: coast to the end, no more decisions
                while not self.done:
                    reward += self._step_env()
                break
            if self.ag.needs_decision(self.obs):
                break
            if k and self.ag.cmd is not None and self.ag.since % k == 0:
                self.fast = True
                break
            if self.pending and any(self.obs["balloon_status"][b, 0] == 2 for _, b, _, _ in self.pending):
                self._resolve()
        self._resolve(final=self.done or not self._burning())
        self._x_last = self.ag.features(self.obs)
        info = {"seed": self.seed_cur, "last_pop": self.last_pop} if self.done else {}
        return self._x_last, reward, self.done, False, info


if __name__ == "__main__":  # smoke test: GNC-like fixed policy (nearest candidate, stretch 1.3)
    import time
    e = GuidanceEnv(); t0 = time.time()
    x, info = e.reset(seed=0); tot, n = 0.0, 0
    while True:
        tok = x[G_DIM:].reshape(N_MAX, F); t1 = np.where(tok[:, 14] > 0, tok[:, 6], 99)
        x, r, d, _, _ = e.step(np.array([int(np.argmin(t1)), 3])); tot += r; n += 1
        if d: break
    print(f"seed {info['seed']}: decisions {n}  pops {tot}  wall {time.time()-t0:.0f}s  obs {x.shape}")

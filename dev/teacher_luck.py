"""Is the search teacher's gain a better route, or luck with this episode's sensor noise? (DECISION_LOG §206)
The teacher searches in exact copies of the simulator, sensor-noise realization included (the balloons and the gust the
agent already predicts from its observation, dev/wind_probe.py). Per seed:
  1. the teacher's search as in dev/s4_teacher.py -> its best branch: launch option + the targets it was forced to
  2. replays of that route, of the policy's own route in the search (its greedy branch, held fixed: the control for
     "a fixed route vs a policy that re-chooses") and of the policy alone, with the episode's own sensor noise ("own": should give back the
     search's numbers) and with NOISE fresh noise draws. The route is forced the way the search forced it: at the first
     leg, after every pop and after LEG_MAX s without one, the next route balloon still up
A route that stays ahead under fresh noise is a choice the policy could learn; one that falls back to the policy was luck.
   .venv/bin/python dev/teacher_luck.py <policy npz> SEED [SEED ...]   (NOISE=4; seeds recorded in the seed registry)"""
import copy, json, logging, os, random, sys, time, warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dev import s4_teacher as T

NOISE = int(os.environ.get("NOISE", 4))


def replay(env0, states, given, kw, route, launch, noise):
    """route None = the policy alone; noise None = the episode's own sensor noise, else a fresh draw from that seed."""
    from dev.search_teacher import rebind
    env = copy.deepcopy(env0); rebind(env)
    np.random.set_state(states[0]); random.setstate(states[1])
    ag = T.make_agent(given, **(kw | ({} if launch is None else dict(launch_at=launch))))
    obs, done, todo, leg, t_f, reseeded = copy.deepcopy(states[2]), False, list(route or []), "first", -1e9, noise is None
    while not done:
        t = float(obs["simulation_time"])
        if todo and ag.p_hat is not None and ag._burning(t) and (
                (leg == "first" and T.at_leg(ag, obs)) or leg == "pop" or (leg == "on" and t - t_f > T.LEG_MAX)):
            st = obs["balloon_status"][:, 0]
            while todo and st[todo[0]] != 1:
                todo.pop(0)
            if todo:
                ag.force, leg, t_f = (todo.pop(0), 0), "on", t
        k0 = T.popped(obs)
        obs, r, te, tr, _ = env.step(ag.get_action(obs))
        done = te or tr
        if not reseeded and env.unwrapped._rocket_flight is not None:
            for j, s in enumerate(env.unwrapped._rocket_flight.sensors):
                s._rng = np.random.default_rng([noise, j])
            reseeded = True
        if T.popped(obs) > k0 and leg == "on":
            leg = "pop"
    return T.popped(obs)


def run(policy, seed):
    from dev import rl2_env
    sl = policy.replace("rl3_policy", "rl2_sl")
    t0 = time.time()
    best, alone, by = T.search(policy, sl, seed)
    route, launch = list(best["ag"].forced), best["launch"]
    params, given = rl2_env.load_scenario_parameters(4)
    env = rl2_env.BalloonPoppingEnv(render_mode=None, parameters=params)
    obs, _ = rl2_env.reset_clean(env, seed)
    states = (np.random.get_state(), random.getstate(), obs)
    kw = dict(policy_path=policy, sl_path=sl)
    groute = best.get("greedy_route")
    out = dict(seed=seed, teacher=T.popped(best["obs"]), alone=alone, by_launch=by, launch=launch, route=route, greedy_route=groute)
    for n in [None] + [[seed, k] for k in range(1, NOISE + 1)]:
        tag = "own" if n is None else f"n{n[1]}"
        out[f"route_{tag}"] = replay(env, states, given, kw, route, launch, n)
        out[f"greedy_{tag}"] = replay(env, states, given, kw, groute, None, n) if groute is not None else None
        out[f"alone_{tag}"] = replay(env, states, given, kw, None, None, n)
    out["wall"] = round(time.time() - t0)
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
    from dev.seed_registry import record
    policy, seeds = sys.argv[1], [int(s) for s in sys.argv[2:]]
    record(seeds, "teacher_luck", note=f"route vs sensor-noise luck (§206) {Path(policy).name}", scenario=4)
    for s in seeds:
        run(policy, s)

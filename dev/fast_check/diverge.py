"""Fly two RL3 policies (A.npz, B.npz, same SL) on one scenario-4 world (reset once, cloudpickled to $FAST_TMP/world_<seed>.pkl for reuse)
in lock step; print the first step where their actions differ and the agents' state there, and each agent's get_action CPU time.
A pickle older than 30 min is useless (its balloon files in /tmp are swept, rl2_env.clean_stale_tmp: the flight ends at once): delete it.
source dev/fast_check/env.sh; $PY dev/fast_check/diverge.py A.npz B.npz SL.npz SEED [k=v (B only) ...]"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import contextlib, copy, io, logging, sys, time, warnings
from pathlib import Path

import cloudpickle
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from BalloonPoppingGymEnv.agents import rl3_agent  # noqa: E402
from BalloonPoppingGymEnv.agents.rl2_sl import load_sl  # noqa: E402
from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv  # noqa: E402
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters  # noqa: E402

assert Path(rl3_agent.__file__).resolve().is_relative_to(ROOT), rl3_agent.__file__
warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)


def world(seed):
    """(env, obs, given) right after env.reset(seed), from the pickle when there is one."""
    f = Path(os.environ.get("FAST_TMP", "$TMPDIR/fast")) / f"world_{seed}.pkl"
    params, given = load_scenario_parameters(4)
    if f.exists():
        env, obs = cloudpickle.loads(f.read_bytes())
    else:
        env = BalloonPoppingEnv(render_mode=None, parameters=params)
        with contextlib.redirect_stdout(io.StringIO()):
            obs, _ = env.reset(seed=seed)
        f.write_bytes(cloudpickle.dumps((env, obs)))
    return env, obs, given


if __name__ == "__main__":
    pa, pb, sl, seed = sys.argv[1:5]
    kw = {k: float(v) for k, v in (r.split("=") for r in sys.argv[5:])}
    env, obs, given = world(int(seed))
    envs, obss = [env, copy.deepcopy(env)], [obs, copy.deepcopy(obs)]
    ags = [rl3_agent.RL3Agent(given, policy_path=pa), rl3_agent.RL3Agent(given, policy_path=pb, **kw)]
    for ag in ags:
        ag.sl = load_sl(sl)
    flat = lambda a: np.array([float(a["launch"]), *np.asarray(a["launch_inclination_heading"], float), *np.asarray(a["tvc"], float),  # noqa: E731
                               float(a["roll"]), float(a["throttle"])])
    n, done, cpu, c0 = 0, False, [0.0, 0.0], time.process_time()
    while not done:
        acts = []
        for i, (ag, o) in enumerate(zip(ags, obss)):
            t0 = time.process_time(); acts.append(ag.get_action(o)); cpu[i] += time.process_time() - t0
        if not np.array_equal(flat(acts[0]), flat(acts[1])):
            print(f"first difference at step {n} (t {float(obss[0]['simulation_time']):.2f} s): {flat(acts[0])} vs {flat(acts[1])}")
            for k in ("since", "cmd", "res", "s_mul", "n_mul", "dv", "dv_p", "dv_v", "target", "stretch", "u_full", "opt_n"):
                print(f"  {k}: {getattr(ags[0], k, None)!r}  |  {getattr(ags[1], k, None)!r}")
            break
        res = [e.step(a) for e, a in zip(envs, acts)]
        obss = [r[0] for r in res]; done = res[0][2] or res[0][3]; n += 1
    else:
        print(f"identical for all {n} steps, pops {res[0][4]['popped_count']} / {res[1][4]['popped_count']}, B counters {ags[1].opt_n}")
    print(f"agent CPU time (get_action, env excluded): A {cpu[0]:.1f} s, B {cpu[1]:.1f} s; env CPU time for both {time.process_time() - c0 - sum(cpu):.1f} s")

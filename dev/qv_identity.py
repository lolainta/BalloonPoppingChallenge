"""Identity check for agents/rl3_q.py: per seed, one env reset, three agents on deep copies of the same world (as
dev/sweep_gnc.py): RL3Agent, RL3Q() defaults, RL3Q(qv_shadow=1, qv_gap=1.0) -> a hash of each episode's full per-step action
sequence; all three must match on every seed (shadow scoring must not change a single action).
  PY=/tmp/llt_py/.venv/bin/python; . dev/node_boot.sh
  $PY dev/qv_identity.py --scenario 40 --seeds 10038736-10038739 --base policy_path=runs/oracle/rl3_policy_cur.npz \
      sl_path=runs/oracle/rl2_sl_cur.npz nav_avg=1e-4 [P.date=2022,7,10,12] [--log runs/qv/ident]
"""
import os, sys, copy, contextlib, io, time, warnings, logging, argparse, hashlib, multiprocessing as mp
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dev.run_seeds import parse_kw, set_param


def run(a):
    seed, sc, base, pkw, log = a
    warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
    from BalloonPoppingGymEnv.agents.rl3_agent import RL3Agent
    from BalloonPoppingGymEnv.agents.rl3_q import RL3Q
    from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv
    from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters
    params, given = load_scenario_parameters(sc)
    for k, v in pkw.items():
        set_param(params, k, v)
    env0 = BalloonPoppingEnv(render_mode=None, parameters=params)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            obs0, _ = env0.reset(seed=seed)
    except Exception as e:
        return seed, repr(e)
    shadow = dict(qv_shadow=1, qv_gap=1.0, qv_seed=seed, qv_cfg=2, **({"qv_log": log} if log else {}))
    out = []
    for mk in (lambda: RL3Agent(given, **base), lambda: RL3Q(given, **base), lambda: RL3Q(given, **base, **shadow)):
        env, obs = copy.deepcopy(env0), copy.deepcopy(obs0)
        ag = mk()
        env._start_wall_time = time.monotonic()  # (as sweep_gnc: the copy's max_wall_time clock restarts)
        h, done, n, t0 = hashlib.sha256(), False, 0, time.monotonic()
        while not done:
            act = ag.get_action(obs)
            for k in sorted(act):
                h.update(k.encode()); h.update(np.asarray(act[k], np.float64).tobytes())
            obs, r, te, tr, info = env.step(act); done = te or tr; n += 1
        out.append((h.hexdigest()[:16], info["popped_count"], n, bool(tr and env.current_step < env.num_timesteps - 1),
                    time.monotonic() - t0, dict(getattr(ag, "opt_n", {}))))
    return seed, out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", type=int, default=40); ap.add_argument("--seeds", default="10038736-10038739")
    ap.add_argument("--base", nargs="*", default=[]); ap.add_argument("--procs", type=int, default=0)
    ap.add_argument("--log", help="qv_log prefix for the shadow agent (also checks that logging changes nothing)")
    a = ap.parse_args()
    lo, _, hi = a.seeds.partition("-")
    seeds = [int(x) for x in a.seeds.split(",")] if "," in a.seeds else list(range(int(lo), int(hi or lo) + 1))
    base = parse_kw(a.base)
    pkw = {k[2:]: base.pop(k) for k in [k for k in base if k.startswith("P.")]}
    from dev.seed_registry import record
    record(seeds, "qv_identity", f"base {' '.join(a.base)}", a.scenario)
    with mp.Pool(min(len(seeds), a.procs or len(os.sched_getaffinity(0)))) as pool:
        res = sorted(pool.map(run, [(s, a.scenario, base, pkw, a.log) for s in seeds]))
    ok = True
    for seed, out in res:
        if isinstance(out, str):
            print(f"seed {seed}: reset failed {out}"); ok = False; continue
        same = len({o[0] for o in out}) == 1
        ok &= same
        print(f"seed {seed}  {'MATCH' if same else 'DIFFER'}  " + "  ".join(
            f"{name} {o[0]} pops {o[1]} steps {o[2]}{' TRUNC' if o[3] else ''} wall {o[4]:.0f}s"
            for name, o in zip(("rl3", "rl3q", "shadow"), out)) + f"  shadow opt_n {out[2][5]}")
    print("ALL MATCH" if ok else "MISMATCH")

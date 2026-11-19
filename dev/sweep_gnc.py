"""Paired GNC / agent-kwarg sweep: one env reset per seed, every config flies a deep copy of the same world
(the scenario-4 reset is ~70% of an episode). Prints per-config mean pops and the paired difference to config 0.
--csv PATH also writes one row per (seed, config): seed, cfg (index), config, pops, last_pop (sim s, empty = none),
steps, truncated (1 = cut by max_wall_time).
AGENT=rl3 .venv/bin/python dev/sweep_gnc.py --scenario 4 --seeds 10000-10095 --base policy_path=... lag_comp=1 \
    --cfg "" "k_att=4" "k_att=4 max_rate=3.5"
AGENT_CLS=rl3q: every config flies RL3Q (agents/rl3_q.py, critic re-ranking probe) instead of RL3Agent.
"""
import os, sys, copy, contextlib, io, time, warnings, logging, argparse, multiprocessing as mp
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dev.run_seeds import parse_kw, set_param
from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters
from rocketpy import Environment

ADD_GUST = Environment.add_wind_gust
def _rec_gust(self, gx, gy):  # remember the gust so that oracle=nogust can take it back out of the rocket's air
    self._gust = (gx, gy); return ADD_GUST(self, gx, gy)
Environment.add_wind_gust = _rec_gust


def oracle(ag, env, modes):
    """Headroom probes (DECISION_LOG §421): hand the agent truth it normally estimates. Harness only, never shipped.
    modes joined by '+': att = true attitude, nav = true attitude + position + velocity, rate = true body rates in the
    attitude loop, wind = the true wind profile for the balloon drift, nogust = the rocket flies without the gust."""
    m = set(modes.split("+"))
    def truth():
        s = np.array(env._rocket_states, float)
        return s if np.isfinite(s).all() else None
    if m & {"att", "nav"}:
        nav0 = ag._navigate
        def nav(acc, pg, vg):
            s = truth()
            if s is not None:
                ag.q = s[6:10].copy()  # INS propagation with the true attitude
            p, v = nav0(acc, pg, vg)
            if s is None:
                return p, v
            ag.q = s[6:10].copy()  # (undo att_fix's nudge)
            if "nav" in m:
                ag.p_hat, ag.v_hat = s[:3].copy(), s[3:6].copy()
            return ag.p_hat, ag.v_hat
        ag._navigate = nav
    if "rate" in m:
        ac0 = ag._attitude_control
        def ac(f, gyro):
            s = truth()
            return ac0(f, gyro if s is None else s[10:13])
        ag._attitude_control = ac
    if "wind" in m:
        e = env._rocketpy_env
        W = np.array([[e.wind_velocity_x(h), e.wind_velocity_y(h)] for h in ag.hz])
        prof = (ag.hz, W, np.vstack([np.zeros((1, 2)), np.cumsum(0.5 * (W[1:] + W[:-1]) * np.diff(ag.hz)[:, None], 0)]))
        def hist(obs):  # same call times as _hist_add, so the profile appears when bwind 4's would
            ag.hprof = prof
            if ag.bwind >= 4:
                ag.wprof = prof
        ag._hist_add = hist
    if "nogust" in m:  # balloons were flown at reset, so only the rocket loses the gust (run_seeds rocket_nogust)
        gx, gy = env._rocketpy_env._gust
        ADD_GUST(env._rocketpy_env, gx * -1, gy * -1)


def run(a):
    seed, sc, base, cfgs, pkw = a
    warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
    cls = os.environ.get("AGENT_CLS") or "rl3"  # rl3q: RL3Q (agents/rl3_q.py), told the seed / config index for its qv_log rows
    if cls == "rl3q":
        from BalloonPoppingGymEnv.agents.rl3_q import RL3Q as Agent
    elif cls == "rl3":
        from BalloonPoppingGymEnv.agents.rl3_agent import RL3Agent as Agent
    else:
        raise ValueError(f"AGENT_CLS={cls!r}: rl3 or rl3q")
    params, given = load_scenario_parameters(sc)
    for k, v in pkw.items():
        set_param(params, k, v)
    env0 = BalloonPoppingEnv(render_mode=None, parameters=params)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            obs0, _ = env0.reset(seed=seed)
    except Exception:
        return seed, None
    out = []
    for j, c in enumerate(cfgs):
        env, obs = copy.deepcopy(env0), copy.deepcopy(obs0)
        kw = {**base, **c}; orc = str(kw.pop("oracle", ""))
        ag = Agent(given, **kw, **(dict(qv_seed=seed, qv_cfg=j) if cls == "rl3q" else {}))
        if orc:
            oracle(ag, env, orc)
        env._start_wall_time = time.monotonic()  # the copy's max_wall_time clock started at env0's reset: restart it
        # (after building the agent: on a node with a slow NFS that alone once took the whole 600 s, 2026-10-02)
        done, last = False, None
        while not done:
            obs, r, te, tr, info = env.step(ag.get_action(obs)); done = te or tr
            if r > 0:
                last = float(obs["simulation_time"])
        wt = bool(tr and env.current_step < env.num_timesteps - 1)
        out.append((info["popped_count"], last, env.current_step, wt))
        if wt:
            print(f"WARNING seed {seed} cfg {c}: truncated by max_wall_time", flush=True)
    return seed, out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", type=int, default=4); ap.add_argument("--seeds", default="10000-10095")
    ap.add_argument("--base", nargs="*", default=[]); ap.add_argument("--cfg", nargs="+", default=[""])
    ap.add_argument("--procs", type=int, default=0); ap.add_argument("--csv"); a = ap.parse_args()
    lo, hi = map(int, a.seeds.split("-")); base = parse_kw(a.base)
    pkw = {k[2:]: base.pop(k) for k in [k for k in base if k.startswith("P.")]}
    cfgs = [parse_kw(c.split()) for c in a.cfg]
    if any(k.startswith("P.") for c in cfgs for k in c):  # a changed world cannot share the reset: one sweep per setting
        sys.exit("P.<param> only works in --base (run one sweep per scenario setting)")
    from dev.seed_registry import record
    record(range(lo, hi + 1), "sweep_gnc", f"base {' '.join(a.base)} cfg {' | '.join(a.cfg)}", a.scenario)
    with mp.Pool(a.procs or len(os.sched_getaffinity(0))) as pool:
        res = [r for r in pool.imap_unordered(run, [(s, a.scenario, base, cfgs, pkw) for s in range(lo, hi + 1)]) if r[1]]
    P = np.array([[o[0] for o in r[1]] for r in res], float)
    if a.csv:
        import csv
        with open(a.csv, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["seed", "cfg", "config", "pops", "last_pop", "steps", "truncated"])
            for seed, out in sorted(res):
                for j, (pops, last, steps, wt) in enumerate(out):
                    w.writerow([seed, j, a.cfg[j], pops, "" if last is None else last, steps, int(wt)])
    print(f"scenario {a.scenario} base {a.base} seeds {len(res)}")
    for j, c in enumerate(a.cfg):
        d = P[:, j] - P[:, 0]
        print(f"  [{c or 'base'}]  mean {P[:, j].mean():.3f}  diff {d.mean():+.3f} +- {d.std() / np.sqrt(len(d)):.3f}")

"""Scenario-4 rollout teacher that never sees the episode's sensor noise (DECISION_LOG §210). dev/s4_teacher.py searched in
exact copies, noise included: ~0.67 of its +1.06 pops was luck with this noise (dev/teacher_luck.py), so its choices
cannot be learned from the observation. Here the episode is flown once, for real, with its own noise; every choice is
made from copies flown with FRESH noise draws:
  - launch: each option in TEACH_LAUNCH (policy / a fixed time) flown to the end under ROLL_M draws
  - at the first leg, after every pop and after LEG_MAX s without one: the policy's top ROLL_K targets, each forced for
    this decision and then the policy alone to the end, under the same ROLL_M draws (common random numbers)
The best mean wins only if it beats the policy's own choice by more than ROLL_MARGIN pops; else the policy's choice.
A one-step rollout improvement of the policy: every label is "better on average", not "lucky here".
Demos as dev/s4_teacher.py, only the rows the teacher chose (legs_only; ROLL_LEGS=0: all) (<out>/<seed>.npz + <run>/teacher_seeds.csv: teacher = this episode's pops, policy_alone = the
policy alone in the same episode, both with the episode's own noise -> an honest per-seed gain).
  .venv/bin/python dev/s4_rollout_teacher.py <run dir or policy npz> <out dir> SEED   (one seed per process)
  .venv/bin/python dev/s4_rollout_teacher.py --legs <full demo dir> <out dir>          (legs_only on saved full demos)"""
import copy, datetime, json, logging, os, random, sys, time, warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dev import s4_teacher as T

K, M = int(os.environ.get("ROLL_K", 4)), int(os.environ.get("ROLL_M", 3))
MARGIN = float(os.environ.get("ROLL_MARGIN", 0.5))
LM = int(os.environ.get("ROLL_LM", M))  # draws per launch option; ROLL_K=0: launch only, no per-leg rollouts (§225)
# ROLL_PAD=1: keep the late pad rows of a changed launch. Off (§216): only "wait" labels, never the policy's own "go" ones
# (dropped with the unchanged launches) -> s4ro launched at ~48 s on 91/96 seeds (was 38 s) and lost 0.2 pops
PAD = int(os.environ.get("ROLL_PAD", 0))


def fly(node, noise=None, until=None):
    """T.step, but sensor noise drawn from `noise` (a SeedSequence entropy list) from the first step the rocket exists."""
    env, ag, obs = node["env"], node["ag"], node["obs"]
    np.random.set_state(node["nrs"]); random.setstate(node["rs"])
    env.unwrapped._start_wall_time = time.monotonic()
    u = env.unwrapped
    while not node["done"]:
        if noise is not None and not node.get("reseeded") and u._rocket_flight is not None:
            for j, s in enumerate(u._rocket_flight.sensors):
                s._rng = np.random.default_rng([*noise, j])
            node["reseeded"] = True
        k0 = T.popped(obs)
        obs, r, te, tr, _ = env.step(ag.get_action(obs))
        if T.popped(obs) > k0 and ag.rec:
            ag.rec[-1][2] += T.popped(obs) - k0
        node["done"] = te or tr
        if until is not None and until(obs):
            break
    node["obs"], node["nrs"], node["rs"] = obs, np.random.get_state(), random.getstate()
    return node


def value(node, seed, tag, force=None, m=None):
    """Mean pops at the end over m (default M) fresh draws, `force` this decision's target first (None: the policy as it is)."""
    out = []
    for m in range(M if m is None else m):
        c = T.clone(node)  # (no "reseeded" key: the copy draws its own noise)
        if force is not None:
            c["ag"].force = (force, 0)
        out.append(T.popped(fly(c, noise=[seed, *tag, m])["obs"]))
    return float(np.mean(out))


def teach(policy, sl, seed):
    from dev import rl2_env
    from dev.search_teacher import rebind
    params, given = rl2_env.load_scenario_parameters(4)
    env = rl2_env.BalloonPoppingEnv(render_mode=None, parameters=params)
    obs, _ = rl2_env.reset_clean(env, seed)
    roots = {}
    for L in T.LAUNCHES:
        ag = T.make_agent(given, **(dict(policy_path=policy, sl_path=sl) | ({} if L is None else dict(launch_at=L))))
        e = copy.deepcopy(env); rebind(e)
        roots[L] = dict(env=e, ag=ag, obs=copy.deepcopy(obs), done=False, nrs=np.random.get_state(), rs=random.getstate())
    alone, x_launch = None, None
    if None in roots:  # the policy, this episode's own noise; its launch decision's features (the last pad row)
        c = fly(T.clone(roots[None])); alone = T.popped(c["obs"])
        pad = [r for r in c["ag"].rec if int(r[1][1]) == 3]
        x_launch = (np.asarray(pad[-1][0], np.float32), float(c["ag"].launched_at)) if pad else None
    vals = {L: value(r, seed, (0,), m=LM) for L, r in roots.items()} if len(roots) > 1 else {L: 0.0 for L in roots}
    base = vals.get(None, -1e9)
    L = max(vals, key=vals.get)
    if L is not None and vals[L] - base <= MARGIN:
        L = None if None in roots else L
    real = roots[L]; ag = real["ag"]; stats = dict(launch_vals=vals, legs=0, overrides=0, x_launch=x_launch)
    if K == 0:  # launch only (§225): the episode as the policy flies it from the chosen launch
        return fly(real), L, alone, stats
    fly(real, until=lambda o: T.at_leg(ag, o))
    leg = 0
    while not real["done"] and ag._burning(float(real["obs"]["simulation_time"])):
        probe = T.clone(real); pa = probe["ag"]; tl, *_ = pa.forward(pa.features(probe["obs"])); del probe
        tgts = [int(i) for i in np.argsort(-tl)[:K] if tl[i] > -1e8]
        if not tgts:
            fly(real, until=lambda o: not ag._burning(float(o["simulation_time"]))); break
        leg += 1
        v = {b: value(real, seed, (1, leg), force=b) for b in tgts}
        b = max(v, key=v.get)
        if v[b] - v[tgts[0]] <= MARGIN:
            b = tgts[0]
        stats["legs"] += 1; stats["overrides"] += int(b != tgts[0])
        ag.force = (b, 0); t0 = float(real["obs"]["simulation_time"]); k0 = T.popped(real["obs"])
        fly(real, until=lambda o: T.popped(o) > k0 or not ag._burning(float(o["simulation_time"]))
            or float(o["simulation_time"]) - t0 > T.LEG_MAX)
    fly(real)  # fly out after the burn (post-burnout pops count)
    return real, L, alone, stats


def legs_only(obs, act, dt, rew, launch_changed):
    """The rows the teacher chose (§213): the first flight decision, every decision right after a pop (both forced by the
    rollouts, overriding or confirming the policy) and the pad rows from 30 s on if it changed the launch. The rest: the policy's own
    outputs (cloning them only copies the policy): dropped, their pops and dt added to the kept row before them."""
    pad = act[:, 1] == 3
    keep = np.zeros(len(act), bool)
    if (~pad).any():
        keep[np.argmax(~pad)] = True
    keep[1:] |= ~pad[1:] & (rew[:-1] > 0)
    if launch_changed:  # ponytail: a fixed 30 s cutoff (the policy launches ~39 s median), not the policy's own launch time
        keep |= pad & (obs[:, 63] * 60.0 >= 30.0)  # x[63] = t / 60; earlier pad rows: both wait, nothing to learn
    g = np.cumsum(keep) - 1; ok = g >= 0; n = int(keep.sum())
    return obs[keep], act[keep], np.bincount(g[ok], dt[ok], n).astype(np.float32), np.bincount(g[ok], rew[ok], n).astype(np.float32)


def filter_dir(src, dst):
    """Apply legs_only to existing demos (src/*.npz -> dst/<same name>), e.g. the evaluation run's full demos."""
    dst = Path(dst); dst.mkdir(parents=True, exist_ok=True)
    for f in sorted(Path(src).glob("[0-9]*.npz")):
        d = dict(np.load(f))
        d["obs"], d["act"], d["dt"], d["rew"] = legs_only(d["obs"], d["act"], d["dt"], d["rew"], float(d["launch"]) >= 0 and PAD)
        tmp = dst / f".{f.name}.tmp.npz"; np.savez(tmp, **d); tmp.replace(dst / f.name)
        print(f.name, len(d["act"]), "rows", float(d["rew"].sum()), "pops")


def main(src, out, seed):
    warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    policy, sl = T.newest(src)
    t0 = time.time()
    try:
        real, L, alone, stats = teach(policy, sl, int(seed))
    except Exception as e:
        print(f"seed {seed} failed: {e!r}", flush=True); return
    rec, pops = real["ag"].rec, T.popped(real["obs"])
    obs = np.array([r[0] for r in rec], np.float32)
    dt = np.append(np.diff(obs[:, 63].astype(np.float64)) * 60.0, 0.5).astype(np.float32)  # x[63] = t / 60 (RL3_FAST)
    act, rew = np.array([r[1] for r in rec], np.float32), np.array([r[2] for r in rec], np.float32)
    if int(os.environ.get("ROLL_LEGS", 1)):
        obs, act, dt, rew = legs_only(obs, act, dt, rew, L is not None and PAD)
    tmp = out / f".{seed}.tmp.npz"
    xl = stats.pop("x_launch")
    np.savez(tmp, obs=obs, act=act, dt=dt, rew=rew, pops=pops, seed=int(seed), policy=policy, launch=-1.0 if L is None else L,
             launch_opts=np.array([-1.0 if k is None else k for k in stats["launch_vals"]]),
             launch_vals=np.array(list(stats["launch_vals"].values())), **({} if xl is None else dict(x_launch=xl[0], t_launch=xl[1])))
    tmp.replace(out / f"{seed}.npz")
    with open(out.parent / "teacher_seeds.csv", "a") as fh:  # every seed whose demo the learner can train on (§201)
        fh.write(f"{int(seed)},{seed}.npz,{pops},{alone},{-1.0 if L is None else L},{Path(policy).name},"
                 f"{datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}\n")
    print(json.dumps(dict(seed=int(seed), teacher=pops, alone=alone, launch=L, rows=len(rec), policy=Path(policy).name,
                          wall=round(time.time() - t0), **stats)), flush=True)


if __name__ == "__main__":
    filter_dir(*sys.argv[2:4]) if sys.argv[1] == "--legs" else main(*sys.argv[1:4])

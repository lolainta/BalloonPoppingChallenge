"""Scenario-4 search teacher (DECISION_LOG §191): per training seed, a beam search over the launch time and the target
order in exact copies of the simulator (dev/gnc2_seq/s4_seq_search: +0.5 pops / episode for the order alone, +0.8 with
the better of two launch times), then the best route's decisions as a demo for the learner's self-imitation
(<run>/demos_in/<seed>.npz, read by dev/async_ppo.py: obs, act, rew, dt; act in the worker's layout, see dev/async_ppo.act).
Every demo's seed is appended to <run>/teacher_seeds.csv (seed, file, teacher pops, policy alone, launch, policy, utc).

  .venv/bin/python dev/s4_teacher.py <run dir or policy npz> <out dir> SEED [SEED ...]   (PROCS=n for a process pool)

The agent is the current RL3Agent (speed gear, residuals, tc / tc2, fast head act as in free flight); a forced target goes
through RL3Agent.choose(obs, tgt=...). Recorded per decision: features, [target, gear, 0, 0, 0, 1, u (the residual mean)]
(pad rows: [0, 3, launch, att u, 1, 0 ...]), and the pops until the next decision. Teacher-only: the agent at evaluation
never searches. Training seeds only (the caller draws them like the workers do and the seed registry records them).
"""
import contextlib, copy, io, json, logging, os, random, sys, time, warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BEAM, TOPK, LEG_MAX = int(os.environ.get("TEACH_BEAM", 4)), int(os.environ.get("TEACH_TOPK", 5)), 10.0
LAUNCHES = [None if x == "policy" else float(x) for x in os.environ.get("TEACH_LAUNCH", "policy,47.5").split(",")]


def popped(obs):
    return int((obs["balloon_status"][:, 0] == 2).sum())


def make_agent(given, **kw):
    from BalloonPoppingGymEnv.agents.rl3_agent import RL3Agent

    class TeachAgent(RL3Agent):
        force = None

        def choose(self, obs, tgt=None):
            if tgt is None and self.force is not None:
                tgt, self.force = self.force[0], None
                self.forced.append(int(tgt))  # the route this branch was forced along (dev/teacher_luck.py replays it)
            return super().choose(obs, tgt)

        def decide(self, obs, a):  # every post-launch decision, after set_res: the row the worker would have sent
            x = self.features(obs)  # before decide: the worker's row carries the features the decision was made on
            super().decide(obs, a)
            u = np.asarray(getattr(self, "u_full", np.zeros(self.n_res)), np.float32)
            self.rec.append([x, np.concatenate([[int(a[0]), int(a[1]), 0, 0.0, 0.0, 1.0], u]), 0.0])

        def decide_launch(self, obs, launch, att=0):  # every pad decision
            x = self.features(obs)
            super().decide_launch(obs, launch, att)
            au = np.asarray(att, float) if np.ndim(att) == 1 else np.zeros(2)
            self.rec.append([x, np.concatenate([[0, 3, int(bool(launch) or float(obs["simulation_time"]) >= 60.0), au[0], au[1], 1.0],
                                                np.zeros(self.n_res)]), 0.0])

    ag = TeachAgent(given, **kw)
    ag.n_res = len(ag.net["res.b"]); ag.rec = []; ag.forced = []
    return ag


def clone(node):
    from dev.search_teacher import rebind
    c = copy.deepcopy({k: node[k] for k in ("env", "ag", "obs", "done", "nrs", "rs")})
    rebind(c["env"])
    return c


def step(node, until=None):
    env, ag, obs = node["env"], node["ag"], node["obs"]
    np.random.set_state(node["nrs"]); random.setstate(node["rs"])
    env.unwrapped._start_wall_time = time.monotonic()  # the 600 s wall clock restarts per advance (§62)
    while not node["done"]:
        k0 = popped(obs)
        obs, r, te, tr, _ = env.step(ag.get_action(obs))
        if popped(obs) > k0 and ag.rec:
            ag.rec[-1][2] += popped(obs) - k0
        node["done"] = te or tr
        if until is not None and until(obs):
            break
    node["obs"], node["nrs"], node["rs"] = obs, np.random.get_state(), random.getstate()
    return node


def at_leg(ag, obs):
    return ag.p_hat is not None and ag._burning(float(obs["simulation_time"])) and (
        ag.cmd is None or obs["balloon_status"][ag.cmd[0], 0] != 1)


def beam(root):
    rank = lambda c: (-popped(c["obs"]), float(c["obs"]["simulation_time"]))
    live, done = [root], []
    while live:
        kids = []
        for node in live:
            if node["done"] or not node["ag"]._burning(float(node["obs"]["simulation_time"])):
                done.append(node); continue
            probe = clone(node); a = probe["ag"]; tl, *_ = a.forward(a.features(probe["obs"])); del probe
            tgts = [int(i) for i in np.argsort(-tl)[:TOPK] if tl[i] > -1e8]
            if not tgts:
                ca = node["ag"]; step(node, until=lambda o: not ca._burning(float(o["simulation_time"]))); done.append(node); continue
            for j, b in enumerate(tgts):
                c = clone(node); ca = c["ag"]; ca.force = (b, 0); t0 = float(c["obs"]["simulation_time"]); k0 = popped(c["obs"])
                step(c, until=lambda o: popped(o) > k0 or not ca._burning(float(o["simulation_time"])) or float(o["simulation_time"]) - t0 > LEG_MAX)
                c["greedy"] = node.get("greedy", False) and j == 0
                kids.append(c); kids.sort(key=rank)
                kids = kids[:BEAM] + [x for x in kids[BEAM:] if x["greedy"]]
        live = kids
    for f in done:
        step(f)  # fly out after the burn (post-burnout pops count)
    return done


def search(policy, sl, seed):
    from dev import rl2_env
    params, given = rl2_env.load_scenario_parameters(4)
    env = rl2_env.BalloonPoppingEnv(render_mode=None, parameters=params)
    obs, _ = rl2_env.reset_clean(env, seed)
    best, alone, by = None, None, {}
    for T in LAUNCHES:
        kw = dict(policy_path=policy, sl_path=sl) | ({} if T is None else dict(launch_at=T))
        ag = make_agent(given, **kw)
        root = dict(env=copy.deepcopy(env) if len(LAUNCHES) > 1 else env, ag=ag, obs=copy.deepcopy(obs), done=False,
                    nrs=np.random.get_state(), rs=random.getstate(), greedy=T is None)
        from dev.search_teacher import rebind
        rebind(root["env"])
        step(root, until=lambda o: at_leg(ag, o))
        fin = beam(root)
        key = lambda f: (popped(f["obs"]), -float(f["obs"]["simulation_time"]))
        b = max(fin, key=key); b["launch"] = T; by["policy" if T is None else f"{T:g}"] = popped(b["obs"])
        if T is None:
            alone = max([popped(f["obs"]) for f in fin if f.get("greedy")] or [-1])
            greedy_route = next((list(f["ag"].forced) for f in fin if f.get("greedy")), None)  # (dev/teacher_luck.py)
        if best is None or key(b) > key(best):
            best = b
    best["greedy_route"] = greedy_route if None in LAUNCHES else None
    return best, alone, by


def newest(run_or_npz):
    p = Path(run_or_npz)
    if p.is_dir():
        p = max(p.glob("rl3_policy_[0-9]*.npz"), key=lambda f: int(f.stem.split("_")[-1]))
    return str(p), str(p).replace("rl3_policy", "rl2_sl")


def main(src, out, seed):
    warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    policy, sl = newest(src)
    t0 = time.time()
    try:
        best, alone, by = search(policy, sl, int(seed))
    except Exception as e:  # e.g. a balloon Monte Carlo that crashes in reset
        print(f"seed {seed} failed: {e!r}", flush=True); return
    rec = best["ag"].rec
    tmp = out / f".{seed}.tmp.npz"  # written, then renamed: the learner never reads a half-written file
    obs = np.array([r[0] for r in rec], np.float32)
    dt = np.append(np.diff(obs[:, 63].astype(np.float64)) * 60.0, 0.5).astype(np.float32)  # x[63] = t / 60: RL3_FAST discounts by time
    np.savez(tmp, obs=obs, act=np.array([r[1] for r in rec], np.float32), dt=dt,
             rew=np.array([r[2] for r in rec], np.float32), pops=popped(best["obs"]), seed=int(seed), policy=policy,
             launch=-1.0 if best["launch"] is None else best["launch"])
    tmp.replace(out / f"{seed}.npz")
    # ledger of every seed whose demo the learner can train on (user 2026-10-16: record them explicitly, rule: every seed
    # the agent met in training is in the seed record; the sbatch also writes the registry before anything flies)
    import datetime
    with open(out.parent / "teacher_seeds.csv", "a") as fh:
        fh.write(f"{int(seed)},{seed}.npz,{popped(best['obs'])},{alone},{-1.0 if best['launch'] is None else best['launch']},"
                 f"{Path(policy).name},{datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}\n")
    print(json.dumps(dict(seed=int(seed), teacher=popped(best["obs"]), alone=alone, by_launch=by, rows=len(rec),
                          policy=Path(policy).name, wall=round(time.time() - t0))), flush=True)


if __name__ == "__main__":
    import multiprocessing as mp
    src, out, seeds = sys.argv[1], sys.argv[2], sys.argv[3:]
    if len(seeds) == 1:  # one seed per process (s4_teacher.sbatch): no Pool, an OOM kill just ends this process (§204)
        sys.exit(main(src, out, seeds[0]))
    with mp.get_context("spawn").Pool(int(os.environ.get("PROCS", 1)), maxtasksperchild=1) as p:
        p.starmap(main, [(src, out, s) for s in seeds])

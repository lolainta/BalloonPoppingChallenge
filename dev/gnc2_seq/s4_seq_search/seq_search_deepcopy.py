"""Sequence-search headroom in scenario 4 (diagnosis only, not a teacher): how many more pops would a better target
ORDER give with our current flight (policy + GNC + eob=2)?

Adapted from dev/search_teacher.py (DECISION_LOG §60 C, §62, §63), nothing there is modified:
  * branch point = the start of a leg (burning and no live commanded target: first choice, or just popped). Children =
    the policy's top-K targets (speed = policy argmax); each child flies the forced target, later decisions follow the
    policy (it may switch mid-flight, as when flying alone) until the next pop / burn end / LEG_MAX s.
  * beam of the BEAM best partial routes by (pops, earlier time) + always the pure-policy child ("greedy").
  * exact copies: deepcopy + rebind of the RocketPy phase solvers (search_teacher.rebind), global numpy / random RNG
    state saved per node and restored before every advance; the env's 600 s wall clock is restarted before every
    advance, so no branch is truncated (each JSON line records whether any truncation happened).
  * free = the plain RL3Agent on the same seed in a fresh process: the greedy route of the search must equal it
    pop-for-pop (exactness check).
Per route: pop times / balloons, hop-start Tmin of the next popped balloon (GNC _reach_time at the moment of the previous
pop, as dev/gnc2_gate/gate.py), bend angle between consecutive pop positions, the rank (in the policy's list) of each
forced choice.

.venv/bin/python dev/gnc2_seq/s4_seq_search/seq_search.py run --seeds 36100-36163 --beam 4 --topk 5 --procs 12 --tag b4k5
.venv/bin/python dev/gnc2_seq/s4_seq_search/seq_search.py report out/b4k5.jsonl
"""
import argparse, contextlib, copy, io, json, logging, os, random, sys, time, warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
HERE = Path(__file__).resolve().parent
CK = os.environ.get("SEARCH_CK") or str(ROOT / "runs/s4bw4/rl3_policy_26342400.npz")
SL = os.environ.get("SEARCH_SL") or str(ROOT / "runs/s4bw4/rl2_sl_26342400.npz")
KW = dict(policy_path=CK, sl_path=SL, eob=2, eob_tau=3, eob_T=4)
for _kv in os.environ.get("SEARCH_KW", "").split():  # extra agent kwargs, e.g. "launch_at=47.5 launch_incl=55 launch_head=255"
    _k, _v = _kv.split("="); KW[_k] = float(_v)
# SEARCH_COND=1 (DECISION_LOG §189): the forced target goes through RL3Agent.choose(obs, tgt=...), so the speed gear,
# residuals (res / res2 / arrival dv, conditioned on the target by tc / tc2) and the fast head act as in free flight;
# the old Teacher path forces (target, gear) and skips them (fine for s4bw4, not for s4fast3)
COND = int(os.environ.get("SEARCH_COND", "0"))


def cond_agent(given, **kw):
    from BalloonPoppingGymEnv.agents.rl3_agent import RL3Agent

    class ForceAgent(RL3Agent):
        force = None
        follow = True

        def choose(self, obs, tgt=None):
            if tgt is None and self.force is not None:
                tgt, self.force = self.force[0], None
            return super().choose(obs, tgt)
    return ForceAgent(given, **kw)
SCEN = 4
LEG_MAX = 10.0


def popped(obs):
    return int((obs["balloon_status"][:, 0] == 2).sum())


def log_pops(node, obs, k0):
    """Record every new pop: (t, balloon, GNC min reach time to all balloons now, p_hat now)."""
    ag = node["ag"]; st = obs["balloon_status"][:, 0]; bs = obs["balloon_states"]
    seen = {b for _, b, _, _ in node["log"]}
    rt = (ag._reach_time(ag.p_hat, ag.v_hat, bs[:, :3], bs[:, 3:]) if ag.p_hat is not None else np.full(len(st), np.inf))
    for b in np.flatnonzero(st == 2):
        if int(b) not in seen:
            node["log"].append((float(obs["simulation_time"]), int(b), rt.astype(np.float32),
                                None if ag.p_hat is None else ag.p_hat.copy()))


def clone(node):
    from dev.search_teacher import rebind
    c = copy.deepcopy({k: node[k] for k in ("env", "ag", "obs", "done", "nrs", "rs", "log", "ranks", "trunc")})
    rebind(c["env"])
    return c


def step(node, until=None):
    env, ag = node["env"], node["ag"]; obs = node["obs"]
    np.random.set_state(node["nrs"]); random.setstate(node["rs"])
    env.unwrapped._start_wall_time = time.monotonic()  # 600 s wall clock -> restart per advance (§62)
    while not node["done"]:
        k0 = popped(obs)
        obs, r, te, tr, _ = env.step(ag.get_action(obs))
        if popped(obs) > k0:
            log_pops(node, obs, k0)
        node["done"] = te or tr
        if tr and float(obs["simulation_time"]) < 59.0:  # wall-clock / other truncation before the sim limit
            node["trunc"] = True
        if until is not None and until(obs):
            break
    node["obs"], node["nrs"], node["rs"] = obs, np.random.get_state(), random.getstate()
    return node


def at_decision(ag, obs):
    return ag.p_hat is not None and ag._burning(float(obs["simulation_time"])) and (
        ag.cmd is None or obs["balloon_status"][ag.cmd[0], 0] != 1)


def beam_search(root, BEAM, TOPK):
    beam, finished, legs = [root], [], 0
    rank = lambda c: (-popped(c["obs"]), float(c["obs"]["simulation_time"]))
    while beam:
        children = []
        for node in beam:
            if node["done"] or not node["ag"]._burning(float(node["obs"]["simulation_time"])):
                finished.append(node); continue
            probe = clone(node)  # features() may touch agent state: never call it on the node that gets cloned
            a = probe["ag"]; tl, sl, *_ = a.forward(a.features(probe["obs"])); del probe
            tgts = [int(i) for i in np.argsort(-tl)[:TOPK] if tl[i] > -1e8]
            s = int(np.argmax(sl))
            if not tgts:  # nothing alive / choosable: fly the node to the end of the burn and keep it
                ca = node["ag"]; step(node, until=lambda o: not ca._burning(float(o["simulation_time"])))
                finished.append(node); continue
            for j, b in enumerate(tgts):
                c = clone(node); ca = c["ag"]
                ca.force = (b, s); t0 = float(c["obs"]["simulation_time"]); k0 = popped(c["obs"])
                c["ranks"] = c["ranks"] + [j]
                step(c, until=lambda o: popped(o) > k0 or not ca._burning(float(o["simulation_time"]))
                     or float(o["simulation_time"]) - t0 > LEG_MAX)
                legs += 1
                c["greedy"] = node["greedy"] and j == 0
                children.append(c)
                children.sort(key=rank)
                children = children[:BEAM] + [x for x in children[BEAM:] if x["greedy"]]
        beam = children
    for f in finished:
        f["ag"].follow = False; step(f)
    return finished, legs


def make_env(seed):
    from dev import rl2_env
    params, given = rl2_env.load_scenario_parameters(SCEN)
    env = rl2_env.BalloonPoppingEnv(render_mode=None, parameters=params)
    obs, _ = rl2_env.reset_clean(env, seed)
    return env, obs, given


def summarize(log, burn_end):
    pops = [(t, b) for t, b, _, _ in log]
    tmin = [float(log[i][2][log[i + 1][1]]) for i in range(len(log) - 1)]
    bend = []
    for i in range(1, len(log) - 1):
        P0, P1, P2 = log[i - 1][3], log[i][3], log[i + 1][3]
        if P0 is None or P2 is None or P1 is None:
            bend.append(None); continue
        u, w = P1 - P0, P2 - P1
        bend.append(float(np.degrees(np.arccos(np.clip(u @ w / max(np.linalg.norm(u) * np.linalg.norm(w), 1e-9), -1, 1)))))
    return dict(pops=[(round(t, 3), b) for t, b in pops], burn_end=burn_end,
                tmin=[x if np.isfinite(x) else None for x in tmin], bend=bend)


def job(args):
    mode, seed, BEAM, TOPK, out = args
    warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
    t0 = time.time()
    try:
        env, obs, given = make_env(seed)
    except Exception as e:
        rec = dict(seed=seed, mode=mode, error=repr(e))
    else:
        if mode == "free":
            from BalloonPoppingGymEnv.agents.rl3_agent import RL3Agent
            ag = RL3Agent(given, **KW)
            node = dict(env=env, ag=ag, obs=obs, done=False, nrs=np.random.get_state(), rs=random.getstate(),
                        log=[], trunc=False)
            step(node)
            be = float(ag.launched_at + ag.burn_time) if ag.launched_at is not None else None
            rec = dict(seed=seed, mode="free", trunc=node["trunc"], **summarize(node["log"], be))
        else:
            from dev.search_teacher import Teacher
            ag = cond_agent(given, **KW) if COND else Teacher(given, **KW); ag.rec = []
            root = dict(env=env, ag=ag, obs=obs, done=False, nrs=np.random.get_state(), rs=random.getstate(),
                        log=[], ranks=[], trunc=False, greedy=True)
            ag.follow = False
            step(root, until=lambda o: at_decision(ag, o))
            ag.follow = True
            fin, legs = beam_search(root, BEAM, TOPK)
            key = lambda f: (popped(f["obs"]), -(f["log"][-1][0] if f["log"] else 1e9))
            best = max(fin, key=key)
            gr = [f for f in fin if f["greedy"]]
            be = lambda f: float(f["ag"].launched_at + f["ag"].burn_time) if f["ag"].launched_at is not None else None
            rec = dict(seed=seed, mode="search", beam=BEAM, topk=TOPK, legs=legs, trunc=any(f["trunc"] for f in fin),
                       best=dict(ranks=best["ranks"], **summarize(best["log"], be(best))),
                       greedy=dict(ranks=gr[0]["ranks"], **summarize(gr[0]["log"], be(gr[0]))) if gr else None,
                       finished_pops=sorted(popped(f["obs"]) for f in fin))
    rec["secs"] = round(time.time() - t0, 1)
    with open(out, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    msg = rec.get("error") or (len(rec["pops"]) if mode == "free" else f"best {len(rec['best']['pops'])} greedy {len(rec['greedy']['pops']) if rec['greedy'] else None}")
    print(f"{mode} seed {seed}: {msg}  {rec['secs']}s", flush=True)


def parse_seeds(s):
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


if __name__ == "__main__":
    import multiprocessing as mp
    p = argparse.ArgumentParser()
    p.add_argument("cmd"); p.add_argument("--seeds", default="36100-36163"); p.add_argument("--beam", type=int, default=4)
    p.add_argument("--topk", type=int, default=5); p.add_argument("--procs", type=int, default=12)
    p.add_argument("--tag", default="b4k5"); p.add_argument("--free", type=int, default=1)
    a = p.parse_args()
    if a.cmd == "run":
        out = HERE / "out" / f"{a.tag}.jsonl"
        seeds = parse_seeds(a.seeds)
        jobs = ([("free", s, 0, 0, str(HERE / "out" / "free.jsonl")) for s in seeds] if a.free else []) + \
               [("search", s, a.beam, a.topk, str(out)) for s in seeds]
        with mp.get_context("spawn").Pool(a.procs, maxtasksperchild=1) as pool:
            list(pool.imap_unordered(job, jobs))

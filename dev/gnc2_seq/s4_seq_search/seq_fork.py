"""Sequence-search headroom in scenario 4, process-per-node version of seq_search_deepcopy.py (same search, same
metrics). A beam node is a live process; branching = os.fork(), so a copy is the whole process image (simulator,
RocketPy solvers, agent, global RNG state) -> exact by construction, no deepcopy / rebind needed. Legs run in parallel
(<= SLOTS flying at once, a shared semaphore; waiting nodes sleep), so one seed's search fits a few minutes of wall time
and the driver can be run in <10 min chunks (resumable: finished seeds are skipped).

Coordinator per seed: node processes connect back over a unix socket, report (pops, sim time, burning, route log), and
wait for a command: expand (fork the policy's top-K forced children, speed = policy argmax, each flies until its next
pop / burn end / LEG_MAX s, later decisions follow the policy), finish (fly out the episode after the burn), or die.
Beam = the BEAM best by (pops, earlier time) + always the pure-policy route (greedy), which must equal the free run.

.venv/bin/python dev/gnc2_seq/s4_seq_search/seq_fork.py --seeds 36100-36163 --beam 4 --topk 5 --tag b4k5 --deadline 560
"""
import argparse, json, logging, multiprocessing as mp, os, sys, tempfile, time, warnings
from multiprocessing.connection import Client, Listener
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import seq_search_deepcopy as sd  # noqa: E402  (KW, SCEN, LEG_MAX, make_env, log_pops, at_decision, summarize, popped)

AUTH = b"seq"
popped = sd.popped


def fly(node, until=None):
    env, ag = node["env"], node["ag"]; obs = node["obs"]
    env.unwrapped._start_wall_time = time.monotonic()  # 600 s wall clock restarted per advance (§62)
    while not node["done"]:
        k0 = popped(obs)
        obs, r, te, tr, _ = env.step(ag.get_action(obs))
        if popped(obs) > k0:
            sd.log_pops(node, obs, k0)
        node["done"] = te or tr
        if tr and float(obs["simulation_time"]) < 59.0:
            node["trunc"] = True
        if until is not None and until(obs):
            break
    node["obs"] = obs


def report(node, final=False):
    ag, obs = node["ag"], node["obs"]
    t = float(obs["simulation_time"])
    be = float(ag.launched_at + ag.burn_time) if ag.launched_at is not None else None
    r = dict(pops=popped(obs), t=t, burning=bool(ag.p_hat is not None and ag._burning(t)), done=bool(node["done"]),
             greedy=node["greedy"], ranks=node["ranks"], trunc=node["trunc"],
             last=node["log"][-1][0] if node["log"] else 1e9)
    if final:
        r["route"] = sd.summarize(node["log"], be)
    return r


def serve(node, addr, sem):
    """Runs inside a node process: report, then obey commands. Never returns (os._exit)."""
    conn = Client(addr, authkey=AUTH)
    conn.send(report(node))
    while True:
        cmd = conn.recv()
        if cmd == "die":
            os._exit(0)
        if cmd == "finish":
            node["ag"].follow = False
            with sem:
                fly(node)
            conn.send(report(node, final=True)); conn.close(); os._exit(0)
        # expand
        ag = node["ag"]
        tl, sl, *_ = ag.forward(ag.features(node["obs"]))  # this process only forks afterwards; children re-derive nothing
        tgts = [int(i) for i in np.argsort(-tl)[:TOPK] if tl[i] > -1e8]
        s = int(np.argmax(sl))
        if not tgts:
            with sem:
                fly(node, until=lambda o: not ag._burning(float(o["simulation_time"])))
                ag.follow = False; fly(node)
            conn.send(("final", report(node, final=True))); conn.close(); os._exit(0)
        conn.send(("expanded", len(tgts))); conn.close()
        for j, b in enumerate(tgts):
            if os.fork() == 0:
                ag.force = (b, s); t0 = float(node["obs"]["simulation_time"]); k0 = popped(node["obs"])
                node["ranks"] = node["ranks"] + [j]; node["greedy"] = node["greedy"] and j == 0
                with sem:
                    fly(node, until=lambda o: popped(o) > k0 or not ag._burning(float(o["simulation_time"]))
                        or float(o["simulation_time"]) - t0 > sd.LEG_MAX)
                serve(node, addr, sem)
        os._exit(0)


def accept(lst):
    c = lst.accept()
    return c, c.recv()


def search_seed(seed, BEAM, sem, out, free_out):
    warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
    t_start = time.time()
    addr = tempfile.mktemp(prefix=f"seq{seed}_", suffix=".sock")
    lst = Listener(addr, family="AF_UNIX", backlog=512, authkey=AUTH)
    lst._listener._socket.settimeout(1800)  # a crashed node would otherwise hang the coordinator
    kids = []
    if free_out and os.fork() == 0:  # the free policy on the same seed (exactness reference)
        with sem:
            try:
                from BalloonPoppingGymEnv.agents.rl3_agent import RL3Agent
                env, obs, given = sd.make_env(seed)
                ag = RL3Agent(given, **sd.KW)
                node = dict(env=env, ag=ag, obs=obs, done=False, log=[], trunc=False)
                fly(node)
                be = float(ag.launched_at + ag.burn_time) if ag.launched_at is not None else None
                rec = dict(seed=seed, mode="free", trunc=node["trunc"], **sd.summarize(node["log"], be))
            except Exception as e:
                rec = dict(seed=seed, mode="free", error=repr(e))
        with open(free_out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        os._exit(0)
    if os.fork() == 0:  # root node
        from dev.search_teacher import Teacher
        with sem:
            try:
                env, obs, given = sd.make_env(seed)
            except Exception as e:
                c = Client(addr, authkey=AUTH); c.send(dict(error=repr(e))); os._exit(0)
            ag = Teacher(given, **sd.KW); ag.rec = []
            node = dict(env=env, ag=ag, obs=obs, done=False, log=[], ranks=[], trunc=False, greedy=True)
            ag.follow = False
            fly(node, until=lambda o: sd.at_decision(ag, o))
            ag.follow = True
        serve(node, addr, sem)
    rec = dict(seed=seed, mode="search", beam=BEAM, topk=TOPK)
    try:
        c, r = accept(lst)
        if "error" in r:
            raise RuntimeError(r["error"])
        beam, finals, legs, levels = [(c, r)], [], 0, 0
        while beam:
            expect = 0
            for c, r in beam:
                if r["done"] or not r["burning"]:
                    c.send("finish"); finals.append(c.recv()); c.close(); continue
                c.send("expand"); tag, x = c.recv(); c.close()
                if tag == "final":
                    finals.append(x)
                else:
                    expect += x
            children = [accept(lst) for _ in range(expect)]
            legs += expect; levels += 1
            children.sort(key=lambda cr: (-cr[1]["pops"], cr[1]["t"], cr[1]["ranks"]))
            keep = children[:BEAM] + [cr for cr in children[BEAM:] if cr[1]["greedy"]]
            for cr in children:
                if not any(cr is k for k in keep):
                    cr[0].send("die"); cr[0].close()
            beam = keep
        best = max(finals, key=lambda f: (f["pops"], -f["last"]))
        gr = [f for f in finals if f["greedy"]]
        rec.update(legs=legs, levels=levels, trunc=any(f["trunc"] for f in finals), best=best,
                   greedy=gr[0] if gr else None, finished_pops=sorted(f["pops"] for f in finals))
    except Exception as e:
        rec["error"] = repr(e)
    lst.close()
    rec["secs"] = round(time.time() - t_start, 1)
    with open(out, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(f"seed {seed}: " + (rec.get("error") or f"best {rec['best']['pops']} greedy {rec['greedy']['pops'] if rec['greedy'] else None} finished {rec['finished_pops']}"
          f" legs {rec['legs']}") + f"  {rec['secs']}s", flush=True)
    while True:  # reap
        try:
            os.waitpid(-1, 0)
        except ChildProcessError:
            break


def parse_seeds(s):
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def done_seeds(path, key="seed"):
    if not Path(path).exists():
        return set()
    return {json.loads(l)[key] for l in open(path) if l.strip()}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="36100-36163"); p.add_argument("--beam", type=int, default=4)
    p.add_argument("--topk", type=int, default=5); p.add_argument("--slots", type=int, default=12)
    p.add_argument("--coords", type=int, default=3); p.add_argument("--tag", default="b4k5")
    p.add_argument("--deadline", type=float, default=560.0, help="s from start; no new seed after deadline - est")
    p.add_argument("--est", type=float, default=300.0); p.add_argument("--free", type=int, default=1)
    a = p.parse_args()
    TOPK = a.topk
    ctx = mp.get_context("fork")
    out = str(HERE / "out" / f"{a.tag}.jsonl"); free_out = str(HERE / "out" / "free.jsonl") if a.free else None
    todo = [s for s in parse_seeds(a.seeds) if s not in done_seeds(out)]
    fdone = done_seeds(free_out) if free_out else set()
    sem = ctx.BoundedSemaphore(a.slots); q = ctx.Queue(); est = ctx.Value("d", a.est)
    for s in todo:
        q.put(s)
    time.sleep(0.2)
    deadline = time.time() + a.deadline
    # free runs only where not yet done (the coordinator gets free_out=None otherwise)
    def coordinator_wrap():
        while True:
            with est.get_lock():
                e = est.value
            if time.time() + e > deadline:
                return
            try:
                seed = q.get(timeout=1)
            except Exception:
                return
            t0 = time.time()
            search_seed(seed, a.beam, sem, out, free_out if seed not in fdone else None)
            with est.get_lock():
                est.value = max(est.value * 0.5 + (time.time() - t0) * 0.5, 60.0)
    procs = [ctx.Process(target=coordinator_wrap) for _ in range(min(a.coords, len(todo)))]
    for pr in procs:
        pr.start()
    for pr in procs:
        pr.join()
    print(f"remaining {len([s for s in parse_seeds(a.seeds) if s not in done_seeds(out)])} seeds", flush=True)

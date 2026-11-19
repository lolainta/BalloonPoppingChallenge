"""Approach-direction headroom in scenario 4 (diagnosis only): how many more pops if the policy could also choose HOW to
approach its own next balloon? Adapted from dev/gnc2_seq/s4_seq_search/seq_fork.py (process-per-node beam, os.fork()
copies, verified exact in dev/gnc2_seq/verify_headroom/); nothing there is modified.

Branch point = start of a leg (burning, no live commanded target). The target is always the policy's own top-1 (speed =
policy argmax); the K children differ only in the APPROACH option of this leg:
  mode appr : option j sets an aim offset d_j; the GNC aims at (balloon + d_j) while the (stretched) time-to-go to the
              balloon itself is > T_sw, then at the balloon (latched off for the leg). d_j fixed at the branch point
              relative to the line of sight L = balloon - p_hat: 0 | +-3 m, +-6 m horizontal perpendicular to L |
              +-4 m vertical | 5 m toward the policy's 2nd-choice balloon (its component perpendicular to L).
  mode chaos: every child flies with no offset; child j > 0 gets p_hat += N(0, EPS) m at the branch point
              (best-of-K inflation control, as verify_headroom/chaos_beam.py).
Beam = the BEAM best by (pops, earlier sim time) + always the all-option-0 route (greedy, must equal the free run).
Per pop we log p_hat, v_hat, the option in force on the leg that ended with it, and all balloon positions, so the
report can compute the angle between the velocity at a pop and the direction to the next popped balloon.

.venv/bin/python dev/gnc2_seq/approach/approach_fork.py --mode appr --tsw 0.8 --seeds 46000-46047 --tag appr08
.venv/bin/python dev/gnc2_seq/approach/approach_fork.py --mode chaos --seeds 46000-46047 --tag chaos
"""
import argparse, json, logging, multiprocessing as mp, os, sys, tempfile, time, warnings
from multiprocessing.connection import Client, Listener
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "dev/gnc2_seq/s4_seq_search"))
import seq_search_deepcopy as sd  # noqa: E402  (make_env: scenario 4, popped, at_decision)

KW = dict(policy_path=str(ROOT / "runs/s4ent/rl3_policy_40915200.npz"),
          sl_path=str(ROOT / "runs/s4ent/rl2_sl_40915200.npz"))  # gnc.* (incl. eob=2) come from the checkpoint
LEG_MAX = 10.0
AUTH = b"appr"
OPTS = ["none", "h+3", "h-3", "h+6", "h-6", "v+4", "v-4", "to2"]
popped = sd.popped


def make_teacher_cls():
    from dev.search_teacher import Teacher

    class ApproachTeacher(Teacher):
        off = None  # dict(tgt, d, tsw, on) for the current leg

        def _guidance(self, p, v, bpos, bvel, t_go):
            o = self.off
            if o is not None and o["on"] and self.target == o["tgt"]:
                if max(t_go * self.stretch, self.t_go_min) > o["tsw"]:
                    bpos = bpos + o["d"]
                else:
                    o["on"] = False
            return super()._guidance(p, v, bpos, bvel, t_go)
    return ApproachTeacher


def offset(ag, obs, j, tgt, second):
    bp = obs["balloon_states"]; L = bp[tgt, :3] - ag.p_hat
    h = np.array([-L[1], L[0], 0.0]); h /= max(np.linalg.norm(h), 1e-9)
    name = OPTS[j]
    if name[0] == "h":
        return float(name[1:]) * h
    if name[0] == "v":
        return np.array([0.0, 0.0, float(name[1:])])
    if name == "to2":
        if second is None:
            return None
        w = bp[second, :3] - bp[tgt, :3]; u = L / max(np.linalg.norm(L), 1e-9); w = w - (w @ u) * u
        return 5.0 * w / max(np.linalg.norm(w), 1e-9)
    return None


def log_pops(node, obs):
    ag = node["ag"]; st = obs["balloon_status"][:, 0]
    seen = {e["b"] for e in node["log"]}
    for b in np.flatnonzero(st == 2):
        if int(b) not in seen:
            node["log"].append(dict(t=float(obs["simulation_time"]), b=int(b), opt=node["opt"],
                                    p=None if ag.p_hat is None else ag.p_hat.copy(),
                                    v=None if ag.v_hat is None else ag.v_hat.copy(),
                                    B=obs["balloon_states"][:, :3].astype(np.float64).copy()))


def summarize(log, burn_end):
    """pops, option on each arrival leg, angle(v at pop i, next popped balloon - p at pop i) in deg, distance."""
    ang, dist = [], []
    for i in range(len(log) - 1):
        e = log[i]
        if e["p"] is None:
            ang.append(None); dist.append(None); continue
        r = e["B"][log[i + 1]["b"]] - e["p"]; v = e["v"]
        ang.append(float(np.degrees(np.arccos(np.clip(r @ v / max(np.linalg.norm(r) * np.linalg.norm(v), 1e-9), -1, 1)))))
        dist.append(float(np.linalg.norm(r)))
    return dict(pops=[(round(e["t"], 3), e["b"]) for e in log], opts=[e["opt"] for e in log], burn_end=burn_end,
                ang=ang, dist=dist, speed=[None if e["v"] is None else float(np.linalg.norm(e["v"])) for e in log])


def burn_end(ag):
    return float(ag.launched_at + ag.burn_time) if ag.launched_at is not None else None


def fly(node, until=None):
    env, ag = node["env"], node["ag"]; obs = node["obs"]
    env.unwrapped._start_wall_time = time.monotonic()  # 600 s wall clock restarted per advance (§62)
    while not node["done"]:
        k0 = popped(obs)
        obs, r, te, tr, _ = env.step(ag.get_action(obs))
        if popped(obs) > k0:
            log_pops(node, obs)
        node["done"] = te or tr
        if tr and float(obs["simulation_time"]) < 59.0:
            node["trunc"] = True
        if until is not None and until(obs):
            break
    node["obs"] = obs


def report(node, final=False):
    ag, obs = node["ag"], node["obs"]
    r = dict(pops=popped(obs), t=float(obs["simulation_time"]), done=bool(node["done"]),
             burning=bool(ag.p_hat is not None and ag._burning(float(obs["simulation_time"]))),
             greedy=node["greedy"], ranks=node["ranks"], trunc=node["trunc"],
             last=node["log"][-1]["t"] if node["log"] else 1e9)
    if final:
        r["route"] = summarize(node["log"], burn_end(ag))
    return r


def serve(node, addr, sem):
    conn = Client(addr, authkey=AUTH)
    conn.send(report(node))
    while True:
        cmd = conn.recv()
        if cmd == "die":
            os._exit(0)
        if cmd == "finish":
            node["ag"].follow = False; node["ag"].off = None
            with sem:
                fly(node)
            conn.send(report(node, final=True)); conn.close(); os._exit(0)
        ag = node["ag"]; obs = node["obs"]
        tl, sl, *_ = ag.forward(ag.features(obs))
        order = [int(i) for i in np.argsort(-tl)[:2] if tl[i] > -1e8]
        s = int(np.argmax(sl))
        if not order:
            ag.off = None
            with sem:
                fly(node, until=lambda o: not ag._burning(float(o["simulation_time"])))
                ag.follow = False; fly(node)
            conn.send(("final", report(node, final=True))); conn.close(); os._exit(0)
        tgt = order[0]; second = order[1] if len(order) > 1 else None
        conn.send(("expanded", K)); conn.close()
        for j in range(K):
            if os.fork() == 0:
                ag.force = (tgt, s); ag.off = None
                if MODE == "appr":
                    d = offset(ag, obs, j, tgt, second)
                    if d is not None:
                        ag.off = dict(tgt=tgt, d=d, tsw=TSW, on=True)
                elif j > 0:
                    rng = np.random.default_rng([node["seed"], len(node["ranks"]), j])
                    ag.p_hat = ag.p_hat + rng.normal(0, EPS, 3)
                t0 = float(obs["simulation_time"]); k0 = popped(obs)
                node["ranks"] = node["ranks"] + [j]; node["greedy"] = node["greedy"] and j == 0; node["opt"] = j
                with sem:
                    fly(node, until=lambda o: popped(o) > k0 or not ag._burning(float(o["simulation_time"]))
                        or float(o["simulation_time"]) - t0 > LEG_MAX)
                ag.off = None
                serve(node, addr, sem)
        os._exit(0)


def accept(lst):
    c = lst.accept()
    return c, c.recv()


def search_seed(seed, BEAM, sem, out, free_out):
    warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
    t_start = time.time()
    addr = tempfile.mktemp(prefix=f"ap{seed}_", suffix=".sock")
    lst = Listener(addr, family="AF_UNIX", backlog=512, authkey=AUTH)
    lst._listener._socket.settimeout(1800)
    if free_out and os.fork() == 0:  # plain RL3Agent on the same seed (exactness reference)
        with sem:
            try:
                from BalloonPoppingGymEnv.agents.rl3_agent import RL3Agent
                env, obs, given = sd.make_env(seed)
                ag = RL3Agent(given, **KW)
                node = dict(env=env, ag=ag, obs=obs, done=False, log=[], trunc=False, opt=0)
                fly(node)
                rec = dict(seed=seed, mode="free", trunc=node["trunc"], **summarize(node["log"], burn_end(ag)))
            except Exception as e:
                rec = dict(seed=seed, mode="free", error=repr(e))
        with open(free_out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        os._exit(0)
    if os.fork() == 0:  # root node
        with sem:
            try:
                env, obs, given = sd.make_env(seed)
            except Exception as e:
                c = Client(addr, authkey=AUTH); c.send(dict(error=repr(e))); os._exit(0)
            ag = make_teacher_cls()(given, **KW); ag.rec = []
            node = dict(env=env, ag=ag, obs=obs, done=False, log=[], ranks=[], trunc=False, greedy=True, opt=0,
                        seed=seed)
            ag.follow = False
            fly(node, until=lambda o: sd.at_decision(ag, o))
            ag.follow = True
        serve(node, addr, sem)
    rec = dict(seed=seed, mode=MODE, beam=BEAM, k=K, tsw=TSW, eps=EPS)
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
    try:
        os.unlink(addr)
    except OSError:
        pass
    rec["secs"] = round(time.time() - t_start, 1)
    with open(out, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(f"[{MODE}] seed {seed}: " + (rec.get("error") or f"best {rec['best']['pops']} greedy {rec['greedy']['pops'] if rec['greedy'] else None}"
          f" finished {rec['finished_pops']} legs {rec['legs']}") + f"  {rec['secs']}s", flush=True)
    while True:
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


def done_seeds(path):
    if not Path(path).exists():
        return set()
    return {json.loads(l)["seed"] for l in open(path) if l.strip()}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("appr", "chaos", "both"), default="appr"); p.add_argument("--tsw", type=float, default=0.8)
    p.add_argument("--eps", type=float, default=0.2); p.add_argument("--k", type=int, default=len(OPTS))
    p.add_argument("--seeds", default="46000-46047"); p.add_argument("--beam", type=int, default=4)
    p.add_argument("--slots", type=int, default=6); p.add_argument("--coords", type=int, default=2)
    p.add_argument("--tag", required=True, help="output tag; with --mode both: APPRTAG,CHAOSTAG"); p.add_argument("--free", type=int, default=1)
    p.add_argument("--deadline", type=float, default=450.0); p.add_argument("--est", type=float, default=300.0)
    a = p.parse_args()
    MODE, TSW, EPS, K = a.mode, a.tsw, a.eps, a.k
    ctx = mp.get_context("fork")
    modes = ["appr", "chaos"] if a.mode == "both" else [a.mode]
    outs = dict(zip(modes, [str(HERE / "out" / f"{t}.jsonl") for t in a.tag.split(",")]))
    free_out = str(HERE / "out" / "free.jsonl") if a.free else None
    done = {m: done_seeds(outs[m]) for m in modes}
    todo = [(m, s) for s in parse_seeds(a.seeds) for m in modes if s not in done[m]]  # paired order: appr, chaos per seed
    sem = ctx.BoundedSemaphore(a.slots); q = ctx.Queue(); est = ctx.Value("d", a.est); fq = ctx.Lock()
    for x in todo:
        q.put(x)
    time.sleep(0.2)
    deadline = time.time() + a.deadline

    def coordinator_wrap():
        global MODE
        while True:
            with est.get_lock():
                e = est.value
            if time.time() + e > deadline:
                return
            try:
                MODE, seed = q.get(timeout=1)
            except Exception:
                return
            t0 = time.time()
            with fq:  # the free reference once per seed
                fo = free_out if (free_out and seed not in done_seeds(free_out)) else None
                if fo:
                    open(fo, "a").write(json.dumps(dict(seed=seed, mode="free_pending")) + "\n")
            search_seed(seed, a.beam, sem, outs[MODE], fo)
            with est.get_lock():
                est.value = max(est.value * 0.5 + (time.time() - t0) * 0.5, 60.0)
    procs = [ctx.Process(target=coordinator_wrap) for _ in range(min(a.coords, len(todo)))]
    for pr in procs:
        pr.start()
    for pr in procs:
        pr.join()
    for m in modes:
        print(f"[{m}] remaining {len([s for s in parse_seeds(a.seeds) if s not in done_seeds(outs[m])])} seeds", flush=True)

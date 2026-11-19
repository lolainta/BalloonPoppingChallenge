"""Chaos control for the sequence-search headroom. Same beam structure as the claimed search (beam B, K children per
branch point, rank by (pops, earlier sim time), keep B + the pure-policy child, fly out after burn), but every child
takes the policy's OWN top-1 target and speed; child j>0 only gets a tiny GNC state perturbation at the branch point
(p_hat += N(0, EPS) m, far below GNSS noise). Any 'gain' here is selection over chaos, not a better sequence.
mode seq: identical machinery but child j forces the policy's rank-j target (independent reimplementation of the claim).
usage: chaos_beam.py MODE TAG EPS BEAM K SLOTS SEED...
"""
import sys, os, json, time, logging, warnings, tempfile
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, os.getcwd())
import numpy as np
import multiprocessing as mp
from multiprocessing.connection import Listener, Client
KW = dict(policy_path="runs/s4bw4/rl3_policy_26342400.npz", sl_path="runs/s4bw4/rl2_sl_26342400.npz", eob=2, eob_tau=3, eob_T=4)
AUTH = b"vh"
MODE, TAG, EPS, BEAM, K, SLOTS = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
SEEDS = [int(s) for s in sys.argv[7:]]
OUT = f"dev/gnc2_seq/verify_headroom/out_{TAG}.jsonl"

def npop(o): return int((o["balloon_status"][:, 0] == 2).sum())

def fly(n, until=None):
    env, ag, obs = n["env"], n["ag"], n["obs"]
    env.unwrapped._start_wall_time = time.monotonic()
    while not n["done"]:
        k0 = npop(obs)
        obs, r, te, tr, _ = env.step(ag.get_action(obs))
        if npop(obs) > k0:
            for b in np.flatnonzero(obs["balloon_status"][:, 0] == 2):
                if int(b) not in n["seen"]:
                    n["seen"].add(int(b)); n["pops"].append([round(float(obs["simulation_time"]), 3), int(b)])
        n["done"] = te or tr
        if tr and float(obs["simulation_time"]) < 59: n["trunc"] = True
        if until and until(obs): break
    n["obs"] = obs

def burning(n): return n["ag"]._burning(float(n["obs"]["simulation_time"]))

def rep(n):
    return dict(pops=npop(n["obs"]), t=float(n["obs"]["simulation_time"]), burning=bool(n["ag"].p_hat is not None and burning(n)),
                done=bool(n["done"]), greedy=n["greedy"], path=n["path"], trunc=n["trunc"], route=n["pops"],
                last=n["pops"][-1][0] if n["pops"] else 1e9)

def serve(n, addr, sem):
    c = Client(addr, authkey=AUTH); c.send(rep(n))
    while True:
        cmd = c.recv()
        if cmd == "die": os._exit(0)
        if cmd == "finish":
            n["ag"].follow = False
            with sem:
                if not n["done"] and burning(n): fly(n, lambda o: not n["ag"]._burning(float(o["simulation_time"])))
                fly(n)
            c.send(rep(n)); c.close(); os._exit(0)
        ag = n["ag"]
        tl, sl, *_ = ag.forward(ag.features(n["obs"]))
        tg = [int(i) for i in np.argsort(-tl)[:K] if tl[i] > -1e8]
        s = int(np.argmax(sl))
        if not tg:
            c.send(("none", 0)); continue
        nk = len(tg) if MODE == "seq" else K
        c.send(("expanded", nk)); c.close()
        for j in range(nk):
            if os.fork() == 0:
                t0 = float(n["obs"]["simulation_time"]); k0 = npop(n["obs"])
                if MODE == "seq":
                    ag.force = (tg[j], s)
                else:
                    ag.force = (tg[0], s)
                    if j > 0:
                        rng = np.random.default_rng([n["seed"], len(n["path"]), j])
                        ag.p_hat = ag.p_hat + rng.normal(0, EPS, 3)
                n["path"] = n["path"] + [j]; n["greedy"] = n["greedy"] and j == 0
                with sem:
                    fly(n, lambda o: npop(o) > k0 or not ag._burning(float(o["simulation_time"])) or float(o["simulation_time"]) - t0 > 10.0)
                serve(n, addr, sem)
        os._exit(0)

def search(seed, sem):
    warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
    t_start = time.time()
    addr = tempfile.mktemp(prefix=f"vh{seed}_", suffix=".sock")
    lst = Listener(addr, family="AF_UNIX", backlog=512, authkey=AUTH)
    if os.fork() == 0:
        from dev import rl2_env
        from dev.search_teacher import Teacher
        with sem:
            params, given = rl2_env.load_scenario_parameters(4)
            env = rl2_env.BalloonPoppingEnv(render_mode=None, parameters=params)
            obs, _ = rl2_env.reset_clean(env, seed)
            ag = Teacher(given, **KW); ag.rec = []
            n = dict(env=env, ag=ag, obs=obs, done=False, seen=set(), pops=[], trunc=False, greedy=True, path=[], seed=seed)
            ag.follow = False
            fly(n, lambda o: ag.p_hat is not None and ag._burning(float(o["simulation_time"])) and (ag.cmd is None or o["balloon_status"][ag.cmd[0], 0] != 1))
            ag.follow = True
        serve(n, addr, sem)
    def acc():
        c = lst.accept(); return c, c.recv()
    beam, finals, legs = [acc()], [], 0
    while beam:
        expect = 0; nxt = []
        for c, r in beam:
            if r["done"] or not r["burning"]:
                c.send("finish"); finals.append(c.recv()); c.close(); continue
            c.send("expand"); tag, x = c.recv()
            if tag == "none":
                c.send("finish"); finals.append(c.recv()); c.close(); continue
            c.close(); expect += x
        ch = [acc() for _ in range(expect)]; legs += expect
        ch.sort(key=lambda cr: (-cr[1]["pops"], cr[1]["t"], cr[1]["path"]))
        keep = ch[:BEAM] + [cr for cr in ch[BEAM:] if cr[1]["greedy"]]
        for cr in ch:
            if not any(cr is k for k in keep): cr[0].send("die"); cr[0].close()
        beam = keep
    lst.close()
    try: os.unlink(addr)
    except OSError: pass
    best = max(finals, key=lambda f: (f["pops"], -f["last"]))
    gr = [f for f in finals if f["greedy"]][0]
    rec = dict(seed=seed, mode=MODE, eps=EPS, beam=BEAM, k=K, legs=legs, best=best["pops"], greedy=gr["pops"],
               best_route=best["route"], greedy_route=gr["route"], best_path=best["path"], trunc=any(f["trunc"] for f in finals),
               finished=sorted(f["pops"] for f in finals), secs=round(time.time() - t_start, 1))
    with open(OUT, "a") as fh: fh.write(json.dumps(rec) + "\n")
    print(seed, "best", rec["best"], "greedy", rec["greedy"], "legs", legs, rec["secs"], "s", flush=True)
    while True:
        try: os.waitpid(-1, 0)
        except ChildProcessError: break

if __name__ == "__main__":
    ctx = mp.get_context("fork")
    sem = ctx.BoundedSemaphore(SLOTS)
    done = set()
    if os.path.exists(OUT): done = {json.loads(l)["seed"] for l in open(OUT)}
    for s in SEEDS:
        if s in done: continue
        p = ctx.Process(target=search, args=(s, sem)); p.start(); p.join()

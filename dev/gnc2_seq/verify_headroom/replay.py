"""Independent checks: (free) fresh-process RL3Agent run; (replay) re-fly the reported best route's rank sequence in
one process, no fork/deepcopy, and compare pop list. Usage: replay.py TAG SEED..."""
import sys, os, json, time, logging, warnings, contextlib, io
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, os.getcwd())
import numpy as np
from multiprocessing import get_context
KW = dict(policy_path="runs/s4bw4/rl3_policy_26342400.npz", sl_path="runs/s4bw4/rl2_sl_26342400.npz", eob=2, eob_tau=3, eob_T=4)
O = "dev/gnc2_seq/s4_seq_search/out/"

def env_for(seed):
    from dev import rl2_env
    params, given = rl2_env.load_scenario_parameters(4)
    env = rl2_env.BalloonPoppingEnv(render_mode=None, parameters=params)
    obs, _ = rl2_env.reset_clean(env, seed)
    return env, obs, given

def npop(o): return int((o["balloon_status"][:, 0] == 2).sum())

def run(env, ag, obs, st, until=None):
    env.unwrapped._start_wall_time = time.monotonic()
    while not st["done"]:
        k0 = npop(obs)
        obs, r, te, tr, _ = env.step(ag.get_action(obs))
        if npop(obs) > k0:
            for b in np.flatnonzero(obs["balloon_status"][:, 0] == 2):
                if int(b) not in st["seen"]:
                    st["seen"].add(int(b)); st["pops"].append([round(float(obs["simulation_time"]), 3), int(b)])
        st["done"] = te or tr
        if tr and float(obs["simulation_time"]) < 59: st["trunc"] = True
        if until and until(obs): break
    return obs

def job(a):
    mode, tag, seed = a
    warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
    t0 = time.time()
    env, obs, given = env_for(seed)
    st = dict(done=False, seen=set(), pops=[], trunc=False)
    if mode == "free":
        from BalloonPoppingGymEnv.agents.rl3_agent import RL3Agent
        ag = RL3Agent(given, **KW); run(env, ag, obs, st)
        return dict(mode=mode, seed=seed, pops=st["pops"], trunc=st["trunc"], secs=time.time() - t0)
    rec = [json.loads(l) for l in open(O + tag + ".jsonl") if json.loads(l)["seed"] == seed][0]
    K = rec["topk"]; ranks = rec["best"]["ranks"]
    from dev.search_teacher import Teacher
    ag = Teacher(given, **KW); ag.rec = []
    ag.follow = False
    burning = lambda o: ag._burning(float(o["simulation_time"]))
    atd = lambda o: ag.p_hat is not None and burning(o) and (ag.cmd is None or o["balloon_status"][ag.cmd[0], 0] != 1)
    obs = run(env, ag, obs, st, atd); ag.follow = True
    for j in ranks:
        assert not st["done"] and burning(obs)
        tl, sl, *_ = ag.forward(ag.features(obs))
        tg = [int(i) for i in np.argsort(-tl)[:K] if tl[i] > -1e8]
        ag.force = (tg[j], int(np.argmax(sl))); t0s = float(obs["simulation_time"]); k0 = npop(obs)
        obs = run(env, ag, obs, st, lambda o: npop(o) > k0 or not burning(o) or float(o["simulation_time"]) - t0s > 10.0)
    if not st["done"] and burning(obs):
        obs = run(env, ag, obs, st, lambda o: not burning(o))
    ag.follow = False; run(env, ag, obs, st)
    return dict(mode=mode, seed=seed, pops=st["pops"], ref=[list(p) for p in rec["best"]["route"]["pops"]], trunc=st["trunc"], secs=time.time() - t0)

if __name__ == "__main__":
    mode, tag = sys.argv[1], sys.argv[2]; seeds = [int(s) for s in sys.argv[3:]]
    free = {json.loads(l)["seed"]: json.loads(l) for l in open(O + "free.jsonl")}
    with get_context("spawn").Pool(min(7, len(seeds)), maxtasksperchild=1) as p:
        for r in p.imap_unordered(job, [(mode, tag, s) for s in seeds]):
            ref = r.get("ref") or [list(x) for x in free[r["seed"]]["pops"]]
            same = [tuple(x) for x in r["pops"]] == [tuple(x) for x in ref]
            print(json.dumps(dict(mode=mode, seed=r["seed"], n=len(r["pops"]), nref=len(ref), identical=same, trunc=r["trunc"], secs=round(r["secs"], 1),
                                  **({} if same else dict(got=r["pops"], ref=ref)))), flush=True)

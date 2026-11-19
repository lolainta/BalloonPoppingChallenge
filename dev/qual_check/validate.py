"""Dry run of a single-file submission through the official evaluator: evaluate.main() on a secret-free temp copy of
dev/qual_check/qual_cfg.yaml (scenario 4, leaderboard_submission false, agent_module_path = AGENT, agent_kwargs = --kw),
one spawned process per seed. The seed is set by wrapping load_scenario_parameters (shared yaml files untouched); the env
class is subclassed only to time reset, hash the actions and note the last pop. Per seed: pops, last-pop time, wall
time reset -> end, max_wall_time hit, and the action-sequence sha256 of the module RL3Agent (policy, SL, the file's
baked kwargs, or --module-kw, + --kw) flown on a deep copy of the same world (copied right after reset: its time counts in the evaluated
env's wall clock, reported as copy). Also checks the embedded policy / SL bytes against --policy / --sl.
.venv/bin/python dev/qual_check/validate.py AGENT.py --policy P.npz --sl SL.npz [--kw "k=v ..."] [--seeds 10000,10001]
    [--procs 4] [--module-kw "k=v ..."] [--no-module]
"""
import os, sys, copy, time, base64, hashlib, tempfile, argparse, multiprocessing as mp
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import numpy as np
import yaml
from dev.run_seeds import parse_kw


def ahash(h, act):
    h.update(np.concatenate([np.asarray(act[k], float).ravel() for k in sorted(act)]).tobytes())


def run(job):
    seed, a = job
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from BalloonPoppingGymEnv.evaluation import evaluate as ev
    kw = parse_kw(a.kw.split())
    orig_load, Env = ev.load_scenario_parameters, ev.BalloonPoppingEnv
    st = {"h": hashlib.sha256(), "last_pop": None, "steps": 0}

    def load(n):
        p, g = orig_load(n)
        p["scenario"]["random_seed"] = seed
        return p, g

    class TimedEnv(Env):
        def reset(self, *ar, **k):
            st["reset_start"] = time.time()
            out = super().reset(*ar, **k)
            st["reset_end"] = time.time()
            if not a.no_module:
                st["copy"] = copy.deepcopy((self, out[0])); st["copy_s"] = time.time() - st["reset_end"]
            return out

        def step(self, action):
            ahash(st["h"], action); st["steps"] += 1
            o, r, te, tr, info = super().step(action)
            if r > 0:
                st["last_pop"] = float(o["simulation_time"])
            if te or tr:
                st["end"] = time.time(); st["pops"] = info["popped_count"]
                st["wall_hit"] = bool(tr and self.current_step < self.num_timesteps - 1)
            return o, r, te, tr, info

    with open(os.path.join(ROOT, "dev/qual_check/qual_cfg.yaml")) as f:
        cfg = yaml.safe_load(f)
    assert cfg["leaderboard_submission"] is False and cfg["scenario_number"] == 4
    cfg.update(agent_module_path=os.path.abspath(a.agent), agent_kwargs=kw, agent_name=f"validate {os.path.basename(a.agent)}")
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(cfg, f)
    ev.load_scenario_parameters, ev.BalloonPoppingEnv = load, TimedEnv
    try:
        env, agent, _ = ev.main([f.name])
    finally:
        os.unlink(f.name)
    g = type(agent).__init__.__globals__  # the single file's module namespace
    emb = [base64.b64decode(g["_WEIGHTS_B64"]) == open(a.policy, "rb").read(),
           base64.b64decode(g["_SL_B64"]) == open(a.sl, "rb").read()]
    res = dict(seed=seed, pops=st["pops"], last_pop=st["last_pop"], steps=st["steps"], wall=st["end"] - st["reset_start"],
               reset=st["reset_end"] - st["reset_start"], wall_hit=st["wall_hit"], hash=st["h"].hexdigest()[:16],
               baked=g.get("_BAKED_KW", {}), emb=emb, copy_s=st.get("copy_s", 0.0))
    if not a.no_module:
        from BalloonPoppingGymEnv.agents.rl3_agent import RL3Agent
        menv, obs = st.pop("copy")
        menv._start_wall_time = time.monotonic()
        _, given = orig_load(4)
        mkw = res["baked"] if a.module_kw is None else parse_kw(a.module_kw.split())
        ag = RL3Agent(given, policy_path=a.policy, sl_path=a.sl, **{**mkw, **kw})
        h, done, mp_ = hashlib.sha256(), False, None
        while not done:
            act = ag.get_action(obs); ahash(h, act)
            obs, r, te, tr, info = menv.step(act); done = te or tr
            if r > 0:
                mp_ = float(obs["simulation_time"])
        res.update(m_hash=h.hexdigest()[:16], m_pops=info["popped_count"], m_last_pop=mp_)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("agent"); ap.add_argument("--policy", required=True); ap.add_argument("--sl", required=True)
    ap.add_argument("--kw", default="", help="agent_kwargs for the eval cfg (and the module agent)")
    ap.add_argument("--seeds", default="10000"); ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--module-kw", help="module agent kwargs instead of the file's baked ones (independent check)")
    ap.add_argument("--no-module", action="store_true"); a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    from dev.seed_registry import record
    record(seeds, "qual_check/validate", f"{os.path.basename(a.agent)} kw {a.kw}", 4)
    with mp.get_context("spawn").Pool(min(a.procs, len(seeds)), maxtasksperchild=1) as pool:
        out = pool.map(run, [(s, a) for s in seeds], chunksize=1)
    ok = True
    for r in out:
        same = r.get("m_hash") == r["hash"]
        ok &= all(r["emb"]) and not r["wall_hit"] and (a.no_module or same)
        print(f"seed {r['seed']} pops {r['pops']} last_pop {r['last_pop']} steps {r['steps']} reset {r['reset']:.0f}s "
              f"reset->end {r['wall']:.0f}s (copy {r['copy_s']:.1f}s) max_wall_time_hit {r['wall_hit']} hash {r['hash']}"
              + ("" if a.no_module else f" | module hash {r['m_hash']} pops {r['m_pops']} last_pop {r['m_last_pop']} "
                 f"identical {same}") + f" | embedded policy/SL == files {r['emb']} baked {r['baked']}", flush=True)
    print("ALL OK" if ok else "CHECK FAILED")
    sys.exit(0 if ok else 1)

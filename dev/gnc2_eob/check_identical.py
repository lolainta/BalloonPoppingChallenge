"""eob=0 must fly exactly like no eob kwarg, and like the gnc_agent.py before the eob change (--orig <copy of it>):
compare the full action sequences on a few seeds (same world). Flies [no kwarg, eob=0] with the current file, and
no kwarg with the --orig file (loaded as BalloonPoppingGymEnv.agents.gnc_agent in its worker processes).
.venv/bin/python dev/gnc2_eob/check_identical.py --scenario 4 --seeds 30000-30003 --orig dev/gnc2_eob/gnc_agent_pre_eob.py \
    --base policy_path=... sl_path=..."""
import os, sys, copy, contextlib, io, warnings, logging, argparse, multiprocessing as mp
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dev.run_seeds import parse_kw
from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters


def run(a):
    seed, sc, base, orig = a
    warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
    if orig:  # fresh (spawned) process: the old GNC module stands in before the agents import it
        import importlib.util
        spec = importlib.util.spec_from_file_location("BalloonPoppingGymEnv.agents.gnc_agent", orig)
        m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m; spec.loader.exec_module(m)
        assert not hasattr(m.GNCAgent, "_eob")
    from BalloonPoppingGymEnv.agents.rl3_agent import RL3Agent
    params, given = load_scenario_parameters(sc)
    env0 = BalloonPoppingEnv(render_mode=None, parameters=params)
    with contextlib.redirect_stdout(io.StringIO()):
        obs0, _ = env0.reset(seed=seed)
    seqs = []
    for c in ([{}] if orig else [{}, {"eob": 0}]):
        env, obs, ag, done, seq = copy.deepcopy(env0), copy.deepcopy(obs0), RL3Agent(given, **{**base, **c}), False, []
        while not done:
            act = ag.get_action(obs)
            seq.append(np.r_[float(act["launch"]), act["launch_inclination_heading"], act["tvc"], act["roll"], act["throttle"]])
            obs, r, te, tr, info = env.step(act); done = te or tr
        seqs.append((np.array(seq), info["popped_count"]))
    return seed, seqs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", type=int, default=4); ap.add_argument("--seeds", default="30000-30003")
    ap.add_argument("--base", nargs="*", default=[]); ap.add_argument("--orig", default=""); a = ap.parse_args()
    lo, hi = map(int, a.seeds.split("-")); seeds = range(lo, hi + 1)
    ctx = mp.get_context("spawn")
    with ctx.Pool(min(8, 2 * len(seeds))) as pool:
        new = pool.map_async(run, [(s, a.scenario, parse_kw(a.base), "") for s in seeds])
        old = pool.map_async(run, [(s, a.scenario, parse_kw(a.base), a.orig) for s in seeds]) if a.orig else None
        new, old = new.get(), (old.get() if old else None)
    for i, (seed, seqs) in enumerate(new):
        runs = seqs + (old[i][1] if old else [])  # [no kwarg, eob=0, (orig no kwarg)]
        same = [A.shape == runs[0][0].shape and np.array_equal(A, runs[0][0]) for A, _ in runs[1:]]
        print(f"scenario {a.scenario} seed {seed}  steps {len(runs[0][0])}  pops {[p for _, p in runs]}  "
              f"eob=0 == no kwarg {same[0]}" + (f"  == orig file {same[1]}" if old else ""))

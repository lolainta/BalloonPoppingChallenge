"""Single-file qualification agent vs module RL3Agent: same world (one reset, deepcopied), full per-step action
sequences compared; plus the single file with eob forced to 0 (must differ) and a count of steps with the
end-of-burn mode engaged (eob_f set).
.venv/bin/python dev/qual_check/equiv.py 10000 10001
(finale: SINGLE=<agent file> POL=<npz> SL=<npz> SC=40 override the qualification defaults)"""
import os, sys, copy, contextlib, io, warnings, logging, time, multiprocessing as mp, importlib.util
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv
from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters

SINGLE = os.environ.get("SINGLE", os.path.join(ROOT, "BalloonPoppingGymEnv/agents/bpc_s4_qual_40915200.py"))
POL = os.environ.get("POL", os.path.join(ROOT, "runs/s4ent/rl3_policy_40915200.npz"))
SL = os.environ.get("SL", os.path.join(ROOT, "runs/s4ent/rl2_sl_40915200.npz"))
SC = int(os.environ.get("SC", 4))


def run(seed):
    warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
    from BalloonPoppingGymEnv.agents.rl3_agent import RL3Agent as Mod
    spec = importlib.util.spec_from_file_location("agent_module", SINGLE)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    params, given = load_scenario_parameters(SC)
    env0 = BalloonPoppingEnv(render_mode=None, parameters=params)
    with contextlib.redirect_stdout(io.StringIO()):
        obs0, _ = env0.reset(seed=seed)
    out = {}
    for name, mk in [("module", lambda: Mod(copy.deepcopy(given), policy_path=POL, sl_path=SL)),
                     ("single", lambda: m.RL3Agent(copy.deepcopy(given))),
                     ("single_eob0", lambda: m.RL3Agent(copy.deepcopy(given), eob=0))]:
        env, obs, ag = copy.deepcopy(env0), copy.deepcopy(obs0), mk()
        done, seq, eob_steps, last_pop, t0 = False, [], 0, None, time.time()
        while not done:
            act = ag.get_action(obs)
            eob_steps += ag.eob_f is not None
            seq.append(np.r_[float(act["launch"]), act["launch_inclination_heading"], act["tvc"], act["roll"], act["throttle"]])
            obs, r, te, tr, info = env.step(act); done = te or tr
            if r > 0:
                last_pop = float(obs["simulation_time"])
        cfg = {k: getattr(ag, k) for k in ("k_att", "max_rate", "coast", "min_alt", "lag_comp", "bwind", "margin", "eob", "eob_tau", "eob_T", "att_fix")}
        out[name] = (np.array(seq), info["popped_count"], last_pop, eob_steps, cfg, time.time() - t0)
    return seed, out


if __name__ == "__main__":
    seeds = [int(s) for s in sys.argv[1:]]
    from dev.seed_registry import record
    record(seeds, "qual_check/equiv", f"{os.path.basename(SINGLE)} single file vs module, eob0", SC)
    with mp.get_context("spawn").Pool(len(seeds)) as pool:
        res = pool.map(run, seeds)
    for seed, out in res:
        A = out["module"][0]
        for name, (S, pops, lp, eob_steps, cfg, wall) in out.items():
            same = S.shape == A.shape and np.array_equal(S, A)
            diff = float(np.max(np.abs(S - A))) if S.shape == A.shape else float("nan")
            print(f"seed {seed} {name:12s} pops {pops:2d} last_pop {lp} steps {len(S)} eob_steps {eob_steps} "
                  f"identical_to_module {same} max|diff| {diff:.3g} wall {wall:.0f}s cfg {cfg}")

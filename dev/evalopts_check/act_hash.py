"""Per-step action hash of deterministic RL3 runs + the eval options' trigger counts (dev/res_check/act_hash.py with
string kwargs, several code trees per call and a code-location check: each job imports the agents from its ROOT, one
process per job, asserted).

.venv/bin/python dev/evalopts_check/act_hash.py JOBFILE [PROCS]
JOBFILE lines: NAME ROOT POLICY.npz SL.npz SEED [k=v ...]
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import contextlib, hashlib, io, logging, multiprocessing as mp, sys, time, warnings

import numpy as np


def run(job):
    name, root, pol, sl, seed, *rest = job.split()
    sys.path.insert(0, root)
    kw = {k: parse(v) for k, v in (r.split("=", 1) for r in rest)}
    warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
    from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv
    from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters
    from BalloonPoppingGymEnv.agents import gnc_agent, rl2_agent, rl3_agent
    from BalloonPoppingGymEnv.agents.rl2_sl import load_sl
    for m in (gnc_agent, rl2_agent, rl3_agent):
        assert os.path.realpath(m.__file__).startswith(os.path.realpath(root) + "/"), m.__file__
    t0 = time.time()
    params, given = load_scenario_parameters(4)
    env = BalloonPoppingEnv(render_mode=None, parameters=params)
    with contextlib.redirect_stdout(io.StringIO()):
        obs, _ = env.reset(seed=int(seed))
    if "gate_keep" in kw:  # reference: dev/gnc2_gate/gate.py's own Fc arm (free policy + must-hit keep r_ok)
        from dev.gnc2_gate.gate import make_agents
        agent = make_agents()[1](given, [], free=1, keep=kw.pop("gate_keep"), policy_path=pol, sl_path=sl, **kw)
        agent.opt_n = agent.n
    else:
        agent = rl3_agent.RL3Agent(given, policy_path=pol, sl_path=sl, **kw)
    agent.sl = load_sl(sl)
    h, n, done, first, pops = hashlib.sha256(), 0, False, None, []
    while not done:
        a = agent.get_action(obs)
        h.update(np.array([float(a["launch"]), *np.asarray(a["launch_inclination_heading"], float), *np.asarray(a["tvc"], float),
                           float(a["roll"]), float(a["throttle"])], np.float64).tobytes())
        obs, r, te, tr, info = env.step(a); n += 1; done = te or tr
        st = obs["balloon_status"][:, 0]
        pops += [(round(float(obs["simulation_time"]), 2), int(b)) for b in np.flatnonzero(st == 2) if int(b) not in {q[1] for q in pops}]
        if first is None and getattr(agent, "opt_n", None):
            first = round(float(obs["simulation_time"]), 2)  # first option trigger
    return (f"{name:14} seed {seed}  hash {h.hexdigest()[:16]}  steps {n}  popped {int(info['popped_count'])}  "
            f"launch {agent.launch_time:.2f}  {' '.join(rest) or 'off'}  counts {dict(getattr(agent, 'opt_n', {}))}  "
            f"first {first}  {time.time() - t0:.0f}s{'  WALL-TRUNCATED' if tr and env.current_step < env.num_timesteps - 1 else ''}\n    pops {pops}")


def parse(v):
    try:
        return float(v)
    except ValueError:
        return v


if __name__ == "__main__":
    jobs = [l.strip() for l in open(sys.argv[1]) if l.strip() and not l.startswith("#")]
    with mp.Pool(min(len(jobs), int(sys.argv[2]) if len(sys.argv) > 2 else 6), maxtasksperchild=1) as p:
        for line in p.imap(run, jobs):
            print(line, flush=True)

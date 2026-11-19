import contextlib, io, os, sys
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
ROOT = './'; sys.path.insert(0, ROOT)
def fly(seed):
    import warnings; warnings.filterwarnings('ignore'); os.chdir(ROOT)
    from BalloonPoppingGymEnv.agents.rl3_agent import RL3Agent
    from dev import rl2_env
    params, given = rl2_env.load_scenario_parameters(4)
    env = rl2_env.BalloonPoppingEnv(render_mode=None, parameters=params)
    with contextlib.redirect_stdout(io.StringIO()):
        obs, _ = env.reset(seed=seed)
    ag = RL3Agent(given, policy_path='runs/s4bw4/rl3_policy_26342400.npz', sl_path='runs/s4bw4/rl2_sl_26342400.npz', eob=2, eob_tau=3, eob_T=4)
    done = False; S = []; t = []
    while not done:
        obs, r, a, b, inf = env.step(ag.get_action(obs)); done = a or b
        S.append(obs['balloon_status'][:, 0].copy()); t.append(float(obs['simulation_time']))
    S = np.array(S); t = np.array(t)
    mine = sorted((round(t[int(np.argmax(S[:, k] == 2))], 2), k) for k in range(100) if (S[:, k] == 2).any())
    z = np.load(f'{ROOT}/dev/gnc2_seq/decomp/data/s4free_{seed}.npz')
    theirs = [(round(a, 2), int(b)) for a, b in z['pops']]
    return seed, len(mine), mine == theirs
if __name__ == '__main__':
    import multiprocessing as mp
    with mp.get_context('spawn').Pool(4) as p:
        print(p.map(fly, [36003, 36017, 36029, 36039]))

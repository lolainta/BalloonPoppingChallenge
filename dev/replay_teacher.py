"""Imitation data from public leaderboard replays: re-run each replay's seed in our env, launch when they launched,
and at every decision target the next balloon of THEIR pop order that is still alive (skipped once we are SLACK s
behind their pop time). GNC flies; the speed level comes from the policy. Records (features, action) per decision
in the RL3 action format -> data/teacher/<name>.npz, and reports how many of their pops our flight reproduces.

.venv/bin/python dev/replay_teacher.py runs/ppo11/rl3_policy_<step>.npz ../lb_replays/*.json
"""
import contextlib, io, json, os, sys
from pathlib import Path

import numpy as np

from BalloonPoppingGymEnv.agents.rl3_agent import RL3Agent
from dev import rl2_env

SLACK = float(os.environ.get("SLACK", 3.0))  # s after their pop time before we give up on that balloon (inf = strict order)
FIX_S = os.environ.get("STRETCH_IDX")  # force one speed level instead of the policy's


def replay_info(path):
    d = json.load(open(path)); tr = d["trajectories"]
    t = np.array([x["time"] for x in tr])
    st = np.array([x["balloon_status"] for x in tr])
    live = [x["rocket_states"] is not None and x["rocket_states"][0] is not None for x in tr]
    pops = sorted((t[np.argmax(st[:, b] == 2)], b) for b in range(st.shape[1]) if (st[:, b] == 2).any())
    return dict(seed=d["meta"]["random_seed"], launch=float(t[int(np.argmax(live))]), pops=pops, score=d["final_reward"],
                b0=np.array(tr[0]["balloon_states"], float))


class Teacher(RL3Agent):
    def __init__(self, given, info, **kw):
        super().__init__(given, **kw)
        self.info, self.X, self.A = info, [], []

    def pick(self, obs, t):
        st = obs["balloon_status"][:, 0]
        for tp, b in self.info["pops"]:
            if st[b] == 1 and t <= tp + SLACK:
                return b
        return None

    def get_action(self, obs):
        t = float(obs["simulation_time"])
        if not self.launched() and self.needs_launch_decision(obs):
            go = t >= self.info["launch"] - 0.25
            self.X.append(self.features(obs)); self.A.append([0, 3, int(go)])
            self.decide_launch(obs, go)
        self.pre_since += 1
        if self.p_hat is not None and self._burning(t) and self.needs_decision(obs):
            x = self.features(obs); tl, sl, *_ = self.forward(x)
            b = self.pick(obs, t)
            a = [b if b is not None else int(np.argmax(tl)), int(FIX_S) if FIX_S else int(np.argmax(sl)), 0]
            self.X.append(x); self.A.append(a + [int(b is not None)])
            self.decide(obs, a)
        self.since += 1
        return super(RL3Agent.__mro__[1], self).get_action(obs)  # GNC


def run(ck, path):
    info = replay_info(path)
    params, given = rl2_env.load_scenario_parameters(1)
    env = rl2_env.BalloonPoppingEnv(render_mode=None, parameters=params)
    with contextlib.redirect_stdout(io.StringIO()):
        obs, _ = env.reset(seed=info["seed"])
    ag = Teacher(given, info, policy_path=ck, sl_path=ck.replace("rl3_policy", "rl2_sl"))
    done, first, n, last = False, True, 0, None
    while not done:
        obs, r, term, trunc, _ = env.step(ag.get_action(obs))
        k = int((obs["balloon_status"][:, 0] == 2).sum())
        if k > n:
            n, last = k, float(obs["simulation_time"])
        if first:  # same balloon field as the replay?
            match = np.abs(obs["balloon_states"][:, :3] - info["b0"][:, :3]).max(); first = False
        done = term or trunc
    st = obs["balloon_status"][:, 0]
    ours = set(np.flatnonzero(st == 2).tolist()); theirs = [b for _, b in info["pops"]]
    name = Path(path).stem
    X, A = np.array(ag.X, np.float32), np.array([a[:3] for a in ag.A])
    np.savez(Path("data/teacher") / f"{name}{os.environ.get('TAG', '')}.npz", X=X, A=A, teacher=np.array([a[3] if len(a) > 3 else 1 for a in ag.A]),
             seed=info["seed"], launch=info["launch"])
    return (f"{name:<14} seed {info['seed']:<8} match {match:.1e}  launch {info['launch']:5.1f}s  their {info['score']:2d}  "
            f"ours {len(ours):2d} ({len(ours & set(theirs))} of theirs)  last pop ours {last or 0:5.1f}s / theirs {info['pops'][-1][0]:5.1f}s  "
            f"decisions {len(X)}")


if __name__ == "__main__":
    import multiprocessing as mp
    ck, files = sys.argv[1], sys.argv[2:]
    with mp.get_context("spawn").Pool(len(files)) as p:
        for line in p.starmap(run, [(ck, f) for f in files]):
            print(line, flush=True)

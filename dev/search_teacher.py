"""Teacher episodes by brute-force search in the real simulator: beam search over the ORDER of targets.
Node = snapshot at a decision point (env + agent + global RNG state + recorded (features, action, reward)).
Expand: the policy's top-K targets x speed levels; each child flies that target until it pops (the policy's
periodic re-decisions keep the same label), the target is gone, or LEG_MAX s -> next node. Keep the BEAM best
nodes by (pops, earlier time) until the burn ends, then fly out the episode. The best path is written as a
self-imitation demo (obs, act, rew) -> <out>/<seed>.npz. One fresh process per seed, so the path reproduces.

Launch-time branches (LAUNCH_TIMES="0,20,30,38"; 0 = the policy's own launch): the pre-launch state is copied,
each branch launches at its time (the waits are recorded as demo decisions too), then gets its own order search;
the best path over all branches is the demo, and the per-branch best is reported.

.venv/bin/python dev/search_teacher.py POLICY.npz OUTDIR SEED [SEED ...]   (env: BEAM, TOPK, FASTK, SPEEDS, LAUNCH_TIMES)
"""
import copy, os, random, sys, time
from pathlib import Path

import numpy as np

from BalloonPoppingGymEnv.agents.rl2_agent import F, N_MAX
from BalloonPoppingGymEnv.agents.rl3_agent import RL3Agent
from dev import rl2_env

BEAM, TOPK, SPEEDS = int(os.environ.get("BEAM", 6)), int(os.environ.get("TOPK", 5)), int(os.environ.get("SPEEDS", 2))
FASTK = int(os.environ.get("FASTK", 2))  # + this many analytically fastest-to-reach targets per node
LEG_MAX = 10.0
LAUNCH_TIMES = [float(x) for x in os.environ.get("LAUNCH_TIMES", "0").split(",")]


class Teacher(RL3Agent):
    """RL3 agent: the first post-launch decision of a leg is forced (self.force = (target, speed)), later ones
    follow the policy (it may switch targets mid-flight, as it does when flying alone). Records every decision."""
    force = None
    follow = True  # False while flying out after the burn
    launch_at = None  # None: the policy decides when to launch; else launch at the first decision >= launch_at

    def get_action(self, obs):
        t = float(obs["simulation_time"])
        if self.net is not None and self.needs_launch_decision(obs):
            x = self.features(obs); _, _, ll, _, al = self.forward(x)
            go = int(np.argmax(ll)) == 1 if self.launch_at is None else t >= self.launch_at - 1e-6
            att = al if "attc.w" in self.net else int(np.argmax(al))  # continuous head: the mean u (2,)
            self.rec.append([x, [0, 3, int(go), *np.atleast_1d(att)], 0.0])
            self.decide_launch(obs, go, att)
        elif not self.launched() and t >= 60.0:
            self.decide_launch(obs, True, getattr(self, "att_last", 0))
        self.pre_since += 1
        if self.p_hat is not None and self._burning(t) and self.needs_decision(obs) and self.follow:
            a = self.force if self.force is not None else self.choose(obs)
            self.force = None
            self.rec.append([self.features(obs), [int(a[0]), int(a[1]), 0, 0], 0.0])
            self.decide(obs, a)
        self.since += 1
        return super(RL3Agent.__mro__[1], self).get_action(obs)


def rebind(env):
    """After deepcopy: point each RocketPy phase solver back at THIS copy's derivative. scipy wraps the ODE
    function in closures, which deepcopy shares, so a copied solver would keep integrating the original flight."""
    from scipy.integrate._ivp.base import check_arguments
    f = getattr(env.unwrapped, "_rocket_flight", None)
    for ph in (getattr(f, "flight_phases", None) or []):
        sol = getattr(ph, "solver", None)
        if sol is None or not hasattr(sol, "_fun"):
            continue
        sol._fun, _ = check_arguments(ph.derivative, sol.y, False)
        sol.fun_single = sol._fun
        def fun(t, y, sol=sol):
            sol.nfev += 1
            return sol.fun_single(t, y)
        sol.fun = fun


def clone(node):
    c = copy.deepcopy({k: node[k] for k in ("env", "ag", "obs", "done", "nrs", "rs")})
    rebind(c["env"])
    return c


def popped(obs):
    return int((obs["balloon_status"][:, 0] == 2).sum())


def step(node, n_max=None, until=None):
    """Advance a node in place; stop at `until(obs)` or episode end. Rewards go to the latest record."""
    env, ag = node["env"], node["ag"]; obs = node["obs"]
    np.random.set_state(node["nrs"]); random.setstate(node["rs"])
    # the env truncates an episode 600 s (wall clock) after reset; copies share that start, and a search spends
    # far longer than that on one seed -> restart the wall clock for every advance (a real episode takes ~20 s)
    env.unwrapped._start_wall_time = time.monotonic()
    n = 0
    while not node["done"]:  # step first: a child starts AT a decision point (its forced choice is taken there)
        k0 = popped(obs)
        obs, r, te, tr, _ = env.step(ag.get_action(obs))
        if popped(obs) > k0 and ag.rec:
            ag.rec[-1][2] += popped(obs) - k0
        node["done"] = te or tr; n += 1
        if (n_max and n >= n_max) or (until is not None and until(obs)):
            break
    node["obs"], node["nrs"], node["rs"] = obs, np.random.get_state(), random.getstate()
    return node


def at_decision(ag, obs):  # the moment a leg starts: burning and no live target (start, or just popped)
    return ag.p_hat is not None and ag._burning(float(obs["simulation_time"])) and (
        ag.cmd is None or obs["balloon_status"][ag.cmd[0], 0] != 1)


def beam_search(root):
    """Order search from a node at the first target choice -> (finished nodes, legs flown).
    Children are pruned as they are made (keep the BEAM best + the pure-policy one): a node is a full copy of the
    simulation (~100 MB), so holding a whole level (BEAM x targets x speeds) would take several GB per process."""
    beam, finished, legs = [root], [], 0
    rank = lambda c: (-popped(c["obs"]), float(c["obs"]["simulation_time"]))
    while beam:
        children = []
        for node in beam:
            if node["done"] or not node["ag"]._burning(float(node["obs"]["simulation_time"])):
                finished.append(node); continue
            a = node["ag"]; x = a.features(node["obs"]); tl, sl, *_ = a.forward(x)
            tok = x[-N_MAX * F:].reshape(N_MAX, F); live = tok[:, 14] > 0.5
            tgts = [int(i) for i in np.argsort(-tl)[:TOPK] if tl[i] > -1e8]  # the policy's favourites first
            # + the analytically fastest to reach: the policy has never flown some states (e.g. after a late
            # launch), so its favourites there can all be out of reach
            tgts += [int(i) for i in np.argsort(np.where(live, tok[:, 6], np.inf))[:FASTK] if live[i] and int(i) not in tgts]
            spds = [int(i) for i in np.argsort(-sl)[:SPEEDS]]
            for b in tgts:
                for s in spds:
                    c = clone(node); ca = c["ag"]
                    ca.force = (b, s); t0 = float(c["obs"]["simulation_time"]); k0 = popped(c["obs"])
                    step(c, until=lambda o: popped(o) > k0 or not ca._burning(float(o["simulation_time"]))
                         or float(o["simulation_time"]) - t0 > LEG_MAX)
                    legs += 1
                    c["greedy"] = node["greedy"] and b == tgts[0] and s == spds[0]
                    children.append(c)  # a leg without a pop stays in: it ranks low, and can still recover
                    children.sort(key=rank)  # prune now: the BEAM best + the pure-policy child
                    children = children[:BEAM] + [x for x in children[BEAM:] if x["greedy"]]
        beam = children
    for f in finished:  # fly out (post-burnout pops)
        f["ag"].follow = False; step(f)
    return finished, legs


def search(policy, seed):
    params, given = rl2_env.load_scenario_parameters(1)
    env = rl2_env.BalloonPoppingEnv(render_mode=None, parameters=params)
    obs, _ = rl2_env.reset_clean(env, seed)
    ag = Teacher(given, policy_path=policy, sl_path=policy.replace("rl3_policy", "rl2_sl")); ag.rec = []
    base = dict(env=env, ag=ag, obs=obs, done=False, nrs=np.random.get_state(), rs=random.getstate())
    key = lambda f: (popped(f["obs"]), -float(f["obs"]["simulation_time"]))
    finished, legs, by_launch, greedy = [], 0, {}, -1  # finished: each branch's best only (nodes are ~100 MB)
    for T in LAUNCH_TIMES:
        root = clone(base) if len(LAUNCH_TIMES) > 1 else base  # on the pad: copying is cheap
        ra = root["ag"]; ra.launch_at = None if T <= 0 else T; ra.follow = False
        root["greedy"] = T <= 0  # "policy alone" = its own launch time and its own choices
        step(root, until=lambda o: at_decision(ra, o))  # fly to the first target choice
        ra.follow = True
        f, n = beam_search(root); legs += n
        bb = max(f, key=key); bb["launch"] = T
        finished.append(bb); by_launch[T] = popped(bb["obs"])
        greedy = max([greedy] + [popped(x["obs"]) for x in f if x["greedy"]])  # the policy flying alone
        del f, root, ra, bb
    return max(finished, key=key), legs, greedy, by_launch


def newest_policy(policy):
    """A run directory -> its newest checkpoint, chosen per seed: the teacher then follows the learner (weights and
    the GNC tunables stored in them) instead of a snapshot taken when the job started."""
    if not Path(policy).is_dir():
        return policy
    return str(max(Path(policy).glob("rl3_policy_[0-9]*.npz"), key=lambda f: int(f.stem.split("_")[-1])))


def main(policy, out, seeds):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    policy = newest_policy(policy)
    for s in seeds:
        t0 = time.time()
        try:
            best, legs, greedy, by_launch = search(policy, int(s))
        except Exception as e:  # e.g. seeds whose balloon Monte Carlo crashes in reset
            print(f"seed {s} failed: {type(e).__name__}", flush=True); continue
        rec = best["ag"].rec
        tmp = out / f".{s}.tmp.npz"  # written then renamed: the learner never reads a half-written file
        np.savez(tmp, obs=np.array([r[0] for r in rec], np.float32), act=np.array([r[1] for r in rec]),
                 rew=np.array([r[2] for r in rec], np.float32), pops=popped(best["obs"]), seed=int(s), policy=policy,
                 launch=best["launch"])
        tmp.replace(out / f"{s}.npz")
        print(f"seed {s} teacher {popped(best['obs'])} pops (policy alone {greedy})  decisions {len(rec)}  legs {legs}  wall {time.time() - t0:.0f}s"
              f"  by launch {' '.join(f'{T:g}:{k}' for T, k in by_launch.items())}", flush=True)


if __name__ == "__main__":
    import multiprocessing as mp
    policy, out, seeds = sys.argv[1], sys.argv[2], sys.argv[3:]
    with mp.get_context("spawn").Pool(int(os.environ.get("PROCS", 1)), maxtasksperchild=1) as p:
        p.starmap(main, [(policy, out, [s]) for s in seeds])

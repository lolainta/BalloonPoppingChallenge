"""Our deterministic run on a leaderboard seed vs the top replays on the same seed (user 2026-09-25 23:00):
launch, pop timeline, altitude / speed at pops, pops before / after burnout, shared balloons.
  .venv/bin/python dev/compare_1957.py POLICY [k=v ...]"""
import contextlib, io, json, sys
import numpy as np
from BalloonPoppingGymEnv.agents.rl3_agent import RL3Agent
from BalloonPoppingGymEnv.agents.rl2_sl import load_sl
from dev import rl2_env
from dev.run_seeds import parse_kw

SEED = 1957083
REPLAYS = ["../lb_replays/18_Taoyuan.json", "../lb_replays/16_SpaceBerry.json", "../lb_replays/13_Apex.json"]


def summary(name, t, pos, vel, status, launch):
    k = (status == 2).sum(1)
    pops = [(t[np.argmax(status[:, b] == 2)], b) for b in range(status.shape[1]) if (status[:, b] == 2).any()]
    pops.sort()
    burn = launch + 30.0
    rows = []
    for tp, b in pops:
        i = int(np.searchsorted(t, tp))
        rows.append((tp - launch, b, pos[i, 2] - 20.0, np.linalg.norm(vel[i])))
    ib = int(np.searchsorted(t, burn)); zmax = np.nanmax(pos[:, 2]) - 20.0
    hops = np.diff([0.0] + [r[0] for r in rows])
    print(f"{name:<22} {len(pops):2d} pops  launch {launch:5.1f}s  last {pops[-1][0] if pops else 0:5.1f}s  "
          f"after burnout {sum(tp > burn for tp, _ in pops)}  max alt {zmax:4.0f} m  burnout alt {pos[min(ib, len(pos)-1), 2]-20:4.0f} m "
          f"v {np.round(vel[min(ib, len(vel)-1)], 1)}  median hop {np.median(hops) if len(hops) else 0:.1f}s")
    print("   pops (s after launch / balloon / alt m / speed m/s): " + "  ".join(f"{r[0]:.1f}/{r[1]}/{r[2]:.0f}/{r[3]:.0f}" for r in rows))
    return set(b for _, b in pops)


def ours(policy, kw):
    params, given = rl2_env.load_scenario_parameters(1)
    env = rl2_env.BalloonPoppingEnv(render_mode=None, parameters=params)
    with contextlib.redirect_stdout(io.StringIO()):
        obs, _ = env.reset(seed=SEED)
    ag = RL3Agent(given, policy_path=policy, **kw)
    if kw.get("sl_path"):
        ag.sl = load_sl(kw["sl_path"])
    T, P, V, S = [], [], [], []
    done = False
    while not done:
        obs, r, te, tr, info = env.step(ag.get_action(obs)); done = te or tr
        rs = np.array(info["rocket_states"][:6], float)
        T.append(float(obs["simulation_time"])); P.append(rs[:3]); V.append(rs[3:]); S.append(obs["balloon_status"][:, 0].copy())
    return np.array(T), np.array(P), np.array(V), np.array(S), ag.launched_at


if __name__ == "__main__":
    pol, kw = sys.argv[1], parse_kw(sys.argv[2:])
    t, p, v, s, L = ours(pol, kw)
    mine = summary("ours " + " ".join(sys.argv[2:])[-30:], t, p, v, s, L)
    for f in REPLAYS:
        d = json.load(open(f)); tr = d["trajectories"]
        tt = np.array([x["time"] for x in tr]); st = np.array([x["balloon_status"] for x in tr]).reshape(len(tr), -1)
        ok = np.array([x["rocket_states"] is not None and x["rocket_states"][0] is not None and not np.isnan(x["rocket_states"][0]) for x in tr])
        rs = np.array([x["rocket_states"][:6] if o else [np.nan] * 6 for x, o in zip(tr, ok)], float)
        theirs = summary(f.split("/")[-1][:-5], tt, rs[:, :3], rs[:, 3:], st, tt[np.argmax(ok)])
        print(f"   shared balloons with ours: {len(mine & theirs)}  (ours only {sorted(mine - theirs)})")

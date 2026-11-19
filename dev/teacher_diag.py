"""Why does our flight miss balloons the leaderboard teams pop? Re-run a replay with its exact target order
(dev/replay_teacher.py, strict), log rocket-target distance every step, and split each chase into passes
(local minima of distance). Report per pass: miss distance, relative speed, and how each chase ended.

.venv/bin/python dev/teacher_diag.py runs/ppo11/rl3_policy_<step>.npz ../lb_replays/[0-9]*_*.json
"""
import contextlib, io, json, os, sys
from pathlib import Path

import numpy as np

os.environ.setdefault("SLACK", "inf")
from dev import replay_teacher as rt
from dev import rl2_env


def their_hit_speeds(path):
    d = json.load(open(path)); tr = d["trajectories"]
    st = np.array([x["balloon_status"] for x in tr]); out = []
    for b in range(st.shape[1]):
        if (st[:, b] == 2).any():
            k = int(np.argmax(st[:, b] == 2)) - 1
            r, bs = tr[k]["rocket_states"], tr[k]["balloon_states"][b]
            if r and r[0] is not None:
                out.append(np.linalg.norm(np.array(r[3:6]) - np.array(bs[3:6])))
    return out


def run(ck, path):
    info = rt.replay_info(path)
    params, given = rl2_env.load_scenario_parameters(1)
    env = rl2_env.BalloonPoppingEnv(render_mode=None, parameters=params)
    with contextlib.redirect_stdout(io.StringIO()):
        obs, _ = env.reset(seed=info["seed"])
    ag = rt.Teacher(given, info, policy_path=ck, sl_path=ck.replace("rl3_policy", "rl2_sl"))
    log = []  # (t, target, dist, rel speed, burning)
    done = False
    while not done:
        obs, r, term, trunc, _ = env.step(ag.get_action(obs))
        done = term or trunc
        t = float(obs["simulation_time"])
        if ag.cmd is not None and ag.p_hat is not None:
            b = ag.cmd[0]; bs = obs["balloon_states"][b]
            log.append((t, b, np.linalg.norm(ag.p_hat - bs[:3]), np.linalg.norm(ag.v_hat - bs[3:]), ag._burning(t),
                        int(obs["balloon_status"][b, 0])))
    burn_end = ag.launched_at + ag.burn_time
    chases = []
    i = 0
    while i < len(log):  # one chase = consecutive steps with the same target
        j = i
        while j + 1 < len(log) and log[j + 1][1] == log[i][1]:
            j += 1
        seg = log[i : j + 1]; d = np.array([s[2] for s in seg])
        mins = [k for k in range(1, len(d) - 1) if d[k] < d[k - 1] and d[k] <= d[k + 1]] + ([len(d) - 1] if len(d) else [])
        passes = [(d[k], seg[k][3]) for k in mins if d[k] < 30]
        end = "popped" if seg[-1][5] == 2 or (j + 1 < len(log) and int(obs["balloon_status"][seg[-1][1], 0]) == 2 and seg[-1][2] < 3) else \
              ("burnout" if not seg[-1][4] else "switched")
        popped = int(obs["balloon_status"][seg[-1][1], 0]) == 2
        chases.append(dict(b=seg[-1][1], t0=seg[0][0], t1=seg[-1][0], popped=popped, passes=passes, min=float(d.min()),
                           burning_end=bool(seg[-1][4])))
        i = j + 1
    return Path(path).stem, info, chases, burn_end, their_hit_speeds(path)


if __name__ == "__main__":
    import multiprocessing as mp
    ck, files = sys.argv[1], sys.argv[2:]
    with mp.get_context("spawn").Pool(len(files)) as p:
        res = p.starmap(run, [(ck, f) for f in files])
    allp, hit_v, miss = [], [], []
    for name, info, chases, burn_end, their_v in res:
        pops = [c for c in chases if c["popped"]]
        print(f"\n== {name} (their {info['score']}, launch {info['launch']:.1f}s, our burnout {burn_end:.1f}s): "
              f"{len(chases)} chases, {len(pops)} popped; their hit speed median {np.median(their_v):.0f} m/s")
        for c in chases:
            ps = " ".join(f"{d:.1f}m@{v:.0f}" for d, v in c["passes"][:5]) or f"never <30 m (min {c['min']:.0f} m)"
            print(f"  b{c['b']:<3} {c['t0']:5.1f}-{c['t1']:5.1f}s ({c['t1'] - c['t0']:4.1f}s) {'POP ' if c['popped'] else 'miss'} passes: {ps}"
                  f"{'' if c['burning_end'] else '  [burnout]'}")
            for d, v in c["passes"]:
                allp.append((d, v, c["popped"]))
            if c["popped"] and c["passes"]:
                hit_v.append(c["passes"][-1][1])
            if not c["popped"]:
                miss.append(c)
        allp_their = their_v
    print("\n== summary")
    n = len(miss)
    near = sum(1 for c in miss if c["min"] < 5); mid = sum(1 for c in miss if 5 <= c["min"] < 30); far = sum(1 for c in miss if c["min"] >= 30)
    burn = sum(1 for c in miss if not c["burning_end"])
    print(f"missed chases {n}: closest <5 m {near}, 5-30 m {mid}, never <30 m {far}; ended after burnout {burn}")
    pz = [p for p in allp if not p[2]]
    print(f"passes on missed balloons: {len(pz)}; miss distance median {np.median([p[0] for p in pz]) if pz else 0:.1f} m")
    print(f"our hit rel. speed median {np.median(hit_v) if hit_v else 0:.0f} m/s")

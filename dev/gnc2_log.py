"""GNC v2 diagnosis data: per-step logs of (a) the leaderboard replays and (b) our flight on the same seeds, so hop timing
can be decomposed offline without re-simulating.

  their_<name>.npz : t (T,), R (T, 13) rocket state [pos, vel, quat e0..e3, body rates] (nan before launch),
                     B (T, 100, 6) balloon pos/vel, S (T, 100) status (0 ground, 1 flying, 2 popped), launch, pops (K, 2) [t, b]
  ours_<mode>_<name>.npz : same arrays for our flight + A (T, 4) action [tvc_x, tvc_y, roll, throttle] (degrees / fraction),
                     F (T, 3) wanted specific force (world, m/s^2), TGT (T,) target, TGO (T,) min reach time of the target (s),
                     STRETCH (T,) speed level multiplier, Q (T, 4) our attitude estimate, a_thrust (T,) thrust accel estimate
     mode strict: their launch time, always their next still-alive balloon (dev/replay_teacher.py with SLACK=inf)
     mode free  : our policy alone (its own launch, targets, speed levels)

.venv/bin/python dev/gnc2_log.py runs/s4bw4/rl3_policy_<step>.npz ../lb_replays/<name>.json ...
"""
import contextlib, io, json, os, sys
from pathlib import Path

os.environ["SLACK"] = "inf"
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from BalloonPoppingGymEnv.agents.rl3_agent import RL3Agent  # noqa: E402
from dev import replay_teacher as rt  # noqa: E402
from dev import rl2_env  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "data/gnc2"


def theirs(path):
    d = json.load(open(path)); tr = d["trajectories"]
    R = np.array([x["rocket_states"] if x["rocket_states"] and x["rocket_states"][0] is not None else [np.nan] * 13 for x in tr], float)
    info = rt.replay_info(path)
    np.savez_compressed(OUT / f"their_{Path(path).stem}.npz", t=np.array([x["time"] for x in tr]), R=R,
                        B=np.array([x["balloon_states"] for x in tr], np.float32), S=np.array([x["balloon_status"] for x in tr], np.int8),
                        launch=info["launch"], pops=np.array(info["pops"], float), seed=info["seed"], score=info["score"])


def ours(ck, path, mode):
    info = rt.replay_info(path)
    params, given = rl2_env.load_scenario_parameters(1)
    env = rl2_env.BalloonPoppingEnv(render_mode=None, parameters=params)
    with contextlib.redirect_stdout(io.StringIO()):
        obs, _ = env.reset(seed=info["seed"])
    kw = dict(policy_path=ck, sl_path=ck.replace("rl3_policy", "rl2_sl"))
    ag = rt.Teacher(given, info, **kw) if mode == "strict" else RL3Agent(given, **kw)
    sel = ag._select_target
    rec = {"tgo": np.nan}
    def select(p, v, o):  # remember the reach time the guidance is given
        b, tg = sel(p, v, o); rec["tgo"] = np.nan if tg is None else tg; return b, tg
    ag._select_target = select
    L = {k: [] for k in "t R B S A F TGT TGO STRETCH Q a_thrust".split()}
    done = False
    while not done:
        a = ag.get_action(obs)
        obs, r, term, trunc, inf = env.step(a)
        done = term or trunc
        dbg = ag.debug or {}
        L["t"].append(float(obs["simulation_time"])); L["R"].append(np.asarray(inf["rocket_states"], float))
        L["B"].append(obs["balloon_states"].astype(np.float32)); L["S"].append(obs["balloon_status"][:, 0].astype(np.int8))
        L["A"].append([*np.asarray(a["tvc"], float), float(a["roll"]), float(a["throttle"])])
        L["F"].append(np.asarray(dbg.get("f", [np.nan] * 3), float)); L["TGT"].append(-1 if dbg.get("target") is None else int(dbg["target"]))
        L["TGO"].append(rec["tgo"]); L["STRETCH"].append(float(ag.stretch)); L["Q"].append(np.asarray(ag.q, float))
        L["a_thrust"].append(float(ag.a_thrust))
    S = np.array(L["S"]); t = np.array(L["t"])
    pops = sorted((t[int(np.argmax(S[:, b] == 2))], b) for b in range(S.shape[1]) if (S[:, b] == 2).any())
    R = np.array(L["R"]); live = ~np.isnan(R[:, 0])
    np.savez_compressed(OUT / f"ours_{mode}_{Path(path).stem}.npz", **{k: np.array(v) for k, v in L.items()},
                        launch=float(t[int(np.argmax(live))]), pops=np.array(pops, float), seed=info["seed"], burn_time=ag.burn_time)
    return f"{mode} {Path(path).stem}: {len(pops)} pops (their {info['score']})"


if __name__ == "__main__":
    import multiprocessing as mp
    import warnings; warnings.filterwarnings("ignore")
    ck, files = sys.argv[1], sys.argv[2:]
    OUT.mkdir(parents=True, exist_ok=True)
    with mp.get_context("spawn").Pool(min(len(files) * 3, os.cpu_count())) as p:
        jobs = [p.apply_async(theirs, (f,)) for f in files]
        res = [p.apply_async(ours, (ck, f, m)) for f in files for m in ("strict", "free")]
        [j.get() for j in jobs]
        for r in res:
            print(r.get())

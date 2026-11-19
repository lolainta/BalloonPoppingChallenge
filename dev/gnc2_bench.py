"""GNC v2 flight benchmark: fly the top teams' replays in their exact order (their launch time, always their next
still-alive balloon; dev/replay_teacher.py with SLACK=inf) with GNC keyword overrides -> pops vs theirs, time between
pops, launch -> first pop. Needs only data/gnc2/replay_info.json (no replay files), so it also runs on the cluster.

.venv/bin/python dev/gnc2_bench.py [--scenario 1] [--procs N] CKPT 'name:k=v,k=v' ['name2:...' ...]
  special key stretch_idx=i forces speed level rl2_agent.STRETCH[i] (0.6, 0.8, 1.0, 1.3)
  e.g. dev/gnc2_bench.py runs/s4bw4/rl3_policy_25478400.npz base: 'katt4:k_att=4,max_rate=3'
"""
import argparse, contextlib, io, json, os, sys
from pathlib import Path

os.environ["SLACK"] = "inf"
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
INFO = ROOT / "data/gnc2/replay_info.json"


def val(v):
    try:
        return float(v)
    except ValueError:
        return v


def run(ck, info, kw, scenario):
    import warnings; warnings.filterwarnings("ignore")
    from dev import replay_teacher as rt
    from dev import rl2_env
    kw = dict(kw)
    if "stretch_idx" in kw:  # force one speed level (index into rl2_agent.STRETCH) instead of the policy's
        rt.FIX_S = str(int(kw.pop("stretch_idx")))
    params, given = rl2_env.load_scenario_parameters(scenario)
    env = rl2_env.BalloonPoppingEnv(render_mode=None, parameters=params)
    with contextlib.redirect_stdout(io.StringIO()):
        obs, _ = env.reset(seed=info["seed"])
    ag = rt.Teacher(given, info, policy_path=ck, sl_path=ck.replace("rl3_policy", "rl2_sl"), **kw)
    done, pops, n = False, [], 0
    while not done:
        obs, r, term, trunc, _ = env.step(ag.get_action(obs))
        done = term or trunc
        k = int((obs["balloon_status"][:, 0] == 2).sum())
        pops += [float(obs["simulation_time"])] * (k - n); n = k
    return pops


if __name__ == "__main__":
    import multiprocessing as mp
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt"); ap.add_argument("variants", nargs="+")
    ap.add_argument("--scenario", type=int, default=1)
    ap.add_argument("--procs", type=int, default=len(os.sched_getaffinity(0)))
    a = ap.parse_args()
    infos = json.loads(INFO.read_text())
    var = []
    for s in a.variants:
        name, _, rest = s.partition(":")
        var.append((name, {k: val(v) for k, v in (x.split("=", 1) for x in rest.split(",") if x)}))
    from dev.seed_registry import record
    record([i["seed"] for i in infos], "gnc2_bench", f"{a.ckpt} {' '.join(a.variants)}", a.scenario)
    tasks = [(a.ckpt, info, kw, a.scenario) for _, kw in var for info in infos]
    with mp.get_context("spawn").Pool(min(a.procs, len(tasks))) as p:
        res = p.starmap(run, tasks)
    their = sum(i["score"] for i in infos)
    print(f"scenario {a.scenario}  ckpt {a.ckpt}  replays {len(infos)} (their total {their})")
    for vi, (name, kw) in enumerate(var):
        r = res[vi * len(infos):(vi + 1) * len(infos)]
        gaps = [g for pt in r for g in np.diff(pt)]
        first = [pt[0] - i["launch"] for pt, i in zip(r, infos) if pt]
        tot = sum(len(pt) for pt in r)
        print(f"{name:14} pops {tot:3} ({100 * tot / their:4.1f}%)  gap median {np.median(gaps) if gaps else float('nan'):.2f}s  "
              f"first pop {np.median(first) if first else float('nan'):.1f}s  per replay {[len(pt) for pt in r]}  {kw}")

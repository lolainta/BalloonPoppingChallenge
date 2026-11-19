"""Strict-order A/B of thrust-utilization variants (same seeds / launch times / target order as data/gnc2/ours_strict_*).

Targets come from THEIR pop order (dev/replay_teacher.Teacher, SLACK = inf), so the policy only picks the speed level;
forcing the level or the throttle is therefore not fighting the policy's target choice.

variants:
  base      : checkpoint settings (should reproduce data/gnc2/ours_strict_*)
  s10       : speed level forced to stretch 1.0 (STRETCH index 2)
  s08       : speed level forced to stretch 0.8 (always saturated: throttle ~1)
  fullsat   : _saturate always scales the net-acceleration command UP to |f| = a_thrust (keeps its direction),
              policy speed level (throttle = 1 whenever a target exists)
  fullsat10 : fullsat + stretch 1.0
  excess2   : excess_dir=2 (unused thrust along the velocity), policy speed level
  noclamp   : _saturate lets the net accel go to -9.3 m/s^2 (f_z >= 0.5) instead of -8.81 (f_z >= 1)

.venv/bin/python dev/gnc2_diag/thrust_util/sim_variants.py base s10 ...   (4 processes)
"""
import contextlib, io, json, os, sys
from pathlib import Path

os.environ["SLACK"] = "inf"
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).resolve().parent / "sims"
CK = str(ROOT / "runs/s4bw4/rl3_policy_25478400.npz")
NAMES = "10_CoolBa 10_SpaceY 11_Mochi 11_near_miss 12_Apex 13_Apex 13_Yaka 14_SpaceBerry 16_SpaceBerry 16_TeamDJ 18_Taoyuan".split()
VARIANTS = {"base": ({}, None, None), "s10": ({}, "2", None), "s08": ({}, "1", None), "fullsat": ({}, None, "full"),
            "fullsat10": ({}, "2", "full"), "excess2": ({"excess_dir": 2}, None, None), "noclamp": ({}, None, "noclamp")}


def make_agent(given, info, variant):
    from dev import replay_teacher as rt
    kw, fix_s, sat = VARIANTS[variant]
    rt.FIX_S = fix_s

    class T(rt.Teacher):
        def _saturate(self, a_cmd):
            if sat is None:
                return super()._saturate(a_cmd)
            g = np.array([0.0, 0.0, 9.81])
            a = a_cmd.copy()
            a[2] = max(a[2], (1.0 if sat == "full" else 0.5) - g[2])
            A, B, C = a @ a, 2 * a @ g, g @ g - self.a_thrust**2
            if sat == "noclamp":
                s = 1.0
                if A > 1e-9 and (a + g) @ (a + g) > self.a_thrust**2:
                    s = max((-B + np.sqrt(max(B * B - 4 * A * C, 0.0))) / (2 * A), 0.0)
                return s * a + g
            if A < 1e-6:  # no command: hover
                return g
            s = max((-B + np.sqrt(max(B * B - 4 * A * C, 0.0))) / (2 * A), 0.0)  # |s a + g| = a_thrust, s may exceed 1
            return s * a + g

    return T(given, info, policy_path=CK, sl_path=CK.replace("rl3_policy", "rl2_sl"), **kw)


def run(name, variant):
    import warnings; warnings.filterwarnings("ignore")
    from dev import replay_teacher as rt
    from dev import rl2_env
    info = rt.replay_info(str(ROOT.parent / f"lb_replays/{name}.json"))
    params, given = rl2_env.load_scenario_parameters(1)
    env = rl2_env.BalloonPoppingEnv(render_mode=None, parameters=params)
    with contextlib.redirect_stdout(io.StringIO()):
        obs, _ = env.reset(seed=info["seed"])
    ag = make_agent(given, info, variant)
    L = {k: [] for k in "t R S A F TGT STRETCH".split()}
    Bl = []
    done = False
    while not done:
        a = ag.get_action(obs)
        obs, r, term, trunc, inf = env.step(a)
        done = term or trunc
        dbg = ag.debug or {}
        L["t"].append(float(obs["simulation_time"])); L["R"].append(np.asarray(inf["rocket_states"], float))
        Bl.append(obs["balloon_states"].astype(np.float32)); L["S"].append(obs["balloon_status"][:, 0].astype(np.int8))
        L["A"].append([*np.asarray(a["tvc"], float), float(a["roll"]), float(a["throttle"])])
        L["F"].append(np.asarray(dbg.get("f", [np.nan] * 3), float)); L["TGT"].append(-1 if dbg.get("target") is None else int(dbg["target"]))
        L["STRETCH"].append(float(ag.stretch))
    S = np.array(L["S"]); t = np.array(L["t"])
    pops = sorted((t[int(np.argmax(S[:, b] == 2))], b) for b in range(S.shape[1]) if (S[:, b] == 2).any())
    R = np.array(L["R"]); live = ~np.isnan(R[:, 0])
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{variant}_{name}.npz", **{k: np.array(v) for k, v in L.items()}, B=np.array(Bl),
                        launch=float(t[int(np.argmax(live))]), pops=np.array(pops, float), seed=info["seed"])
    lt = float(t[int(np.argmax(live))])
    pt = np.array([p[0] for p in pops])
    burn_pops = int((pt <= lt + 30.0).sum())
    return dict(name=name, variant=variant, pops=len(pops), pops_burn=burn_pops,
                hops=list(np.diff(np.concatenate([[lt], pt])).round(3)))


if __name__ == "__main__":
    import multiprocessing as mp
    variants = sys.argv[1:] or ["base"]
    jobs = [(n, v) for v in variants for n in NAMES]
    res = []
    with mp.get_context("spawn").Pool(4) as p:
        for r in p.starmap(run, jobs):
            res.append(r)
            print(r["variant"], r["name"], r["pops"], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    for v in variants:
        rs = [r for r in res if r["variant"] == v]
        json.dump(rs, open(OUT / f"summary_{v}.json", "w"), indent=1, default=float)
        hops = np.concatenate([np.array(r["hops"][1:], float) for r in rs if len(r["hops"]) > 1])
        first = np.array([r["hops"][0] for r in rs if r["hops"]], float)
        print(f"== {v}: pops {sum(r['pops'] for r in rs)} (in burn {sum(r['pops_burn'] for r in rs)}), "
              f"median hop {np.median(hops):.2f} mean {hops.mean():.2f}, first pop median {np.median(first):.2f}")

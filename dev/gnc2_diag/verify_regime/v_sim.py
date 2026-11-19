"""Independent strict re-runs (own subclass, no import of launch_regime code).
variants:
  instr     : unchanged flight; logs every _saturate call: clamp bound (a_cmd_z < 1 - g) and the scale s -> true clamp fraction
              incl. the saturated case that the logged 'F_z == 1.0' test misses
  att+2/att-2 : THEIR launch attitude with the heading rotated by +2 / -2 deg (chaos around the claimed +10)
  low68+2   : 68 deg toward the low balloons, heading +2 deg
.venv/bin/python dev/gnc2_diag/verify_regime/v_sim.py <variant> [names...]
"""
import contextlib, io, os, sys
from pathlib import Path
os.environ["SLACK"] = "inf"
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from BalloonPoppingGymEnv.agents.gnc_agent import get_initial_attitude, quat_to_R  # noqa: E402
from dev import replay_teacher as rt  # noqa: E402
from dev import rl2_env  # noqa: E402

CK = str(ROOT / "runs/s4bw4/rl3_policy_25478400.npz")
OUT = Path(__file__).resolve().parent / "sim"
NAMES = "10_CoolBa 10_SpaceY 11_Mochi 11_near_miss 12_Apex 13_Apex 13_Yaka 14_SpaceBerry 16_SpaceBerry 16_TeamDJ 18_Taoyuan".split()


def their_zb(name):
    d = np.load(ROOT / f"data/gnc2/their_{name}.npz"); R = d["R"]
    q = R[int(np.flatnonzero(np.isfinite(R[:, 0]))[0]), 6:10]
    return quat_to_R(q)[:, 2]


def att_from_zb(z):
    """(incl, heading) in the env convention: the grid heading whose get_initial_attitude body z best matches z."""
    incl = float(np.degrees(np.arcsin(np.clip(z[2], -1, 1))))
    hs = np.arange(0, 360, 0.25)
    best = max(hs, key=lambda h: quat_to_R(np.array(get_initial_attitude(incl, float(h))))[:, 2] @ z)
    return incl, float(best)


def run(variant, name):
    path = str(ROOT.parent / "lb_replays" / f"{name}.json")
    info = rt.replay_info(path)
    params, given = rl2_env.load_scenario_parameters(1)
    env = rl2_env.BalloonPoppingEnv(render_mode=None, parameters=params)
    with contextlib.redirect_stdout(io.StringIO()):
        obs, _ = env.reset(seed=info["seed"])
    kw = dict(policy_path=CK, sl_path=CK.replace("rl3_policy", "rl2_sl"))
    if variant.startswith("low68"):
        kw["launch_incl"] = 68.0
    log = []

    class T(rt.Teacher):
        def decide_launch(self, obs, launch, att=0):
            super().decide_launch(obs, launch, att)
            if self.launched() and variant != "instr":
                if variant.startswith("att"):
                    incl, head = att_from_zb(their_zb(name)); head += float(variant[3:])
                else:
                    incl, head = self.incl_head[0], self.incl_head[1] + float(variant[5:])
                self.incl_head = np.array([incl, head % 360])
                self.q = np.array(get_initial_attitude(*self.incl_head))

        def _saturate(self, a_cmd):
            f = super()._saturate(a_cmd)
            if variant == "instr":
                a = a_cmd.copy(); a[2] = max(a[2], 1.0 - 9.81)
                g = np.array([0, 0, 9.81]); sat = (a + g) @ (a + g) > self.a_thrust**2
                log.append((self.launched_at is not None and float(self.burn_left), float(a_cmd[2]) < 1.0 - 9.81, bool(sat), float(f[2])))
            return f

    ag = T(given, info, **kw)
    done, Tt, S, R = False, [], [], []
    while not done:
        a = ag.get_action(obs)
        obs, r, term, trunc, inf = env.step(a)
        done = term or trunc
        Tt.append(float(obs["simulation_time"])); S.append(obs["balloon_status"][:, 0].astype(np.int8)); R.append(np.asarray(inf["rocket_states"], float))
    S, Tt, R = np.array(S), np.array(Tt), np.array(R)
    pops = sorted((Tt[int(np.argmax(S[:, b] == 2))], b) for b in range(100) if (S[:, b] == 2).any())
    tl = Tt[int(np.flatnonzero(np.isfinite(R[:, 0]))[0])]
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT / f"{variant}_{name}.npz", pops=np.array(pops, float).reshape(-1, 2), launch=tl, incl_head=ag.incl_head,
                        log=np.array(log, float).reshape(-1, 4))
    p = np.array([x[0] for x in pops]) - tl
    return variant, name, len(pops), (p[0] if len(p) else np.nan), tuple(np.round(ag.incl_head, 2))


if __name__ == "__main__":
    import multiprocessing as mp, warnings
    warnings.filterwarnings("ignore")
    variant = sys.argv[1]; names = sys.argv[2:] or NAMES
    with mp.get_context("spawn").Pool(4) as pool:
        res = pool.starmap(run, [(variant, n) for n in names])
    for r in res:
        print(f"{r[0]:8s} {r[1]:15s} pops {r[2]:2d} first {r[3]:5.2f} att {r[4]}")
    print(f"== {variant}: total {sum(r[2] for r in res)}", flush=True)

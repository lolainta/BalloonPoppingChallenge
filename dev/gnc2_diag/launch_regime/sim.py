"""Strict-order / free re-runs with launch-attitude and regime-rule variants (no existing file is modified: variants are
subclasses / constructor kwargs). Writes dev/gnc2_diag/launch_regime/sim/<variant>_<mode>_<name>.npz and prints a summary.

.venv/bin/python dev/gnc2_diag/launch_regime/sim.py <variant>[,<variant>...] [strict|free] [names...]
variants:
  base      : as data/gnc2 (sanity check)
  att       : strict only - launch with THEIR initial attitude (inclination / heading from their first quaternion)
  low68     : launch at 68 deg toward the balloons below 10 m (existing launch_incl override, heading = _low_heading)
  low75     : same at 75 deg
  nodown    : "never thrust downward" clamp relaxed: vertical specific force >= -3 m/s^2 instead of >= +1 m/s^2
  vert      : free only - force a vertical launch
  know      : strict only - THEIR attitude + foreknowledge: while their first balloon is still on the ground, GNC and the
              teacher see it as a live stationary balloon 2 m above its ground spot (a phantom; the env is untouched)
  j<h>      : nuisance jitter for the noise floor - launch at 89.5 deg, heading h (e.g. j0, j90, j180, j270)
"""
import contextlib, io, os, sys
from pathlib import Path

os.environ["SLACK"] = "inf"
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from BalloonPoppingGymEnv.agents.gnc_agent import get_initial_attitude  # noqa: E402
from BalloonPoppingGymEnv.agents.rl3_agent import RL3Agent  # noqa: E402
from dev import replay_teacher as rt  # noqa: E402
from dev import rl2_env  # noqa: E402

CK = str(ROOT / "runs/s4bw4/rl3_policy_25478400.npz")
OUT = Path(__file__).resolve().parent / "sim"
REPLAYS = ROOT.parent / "lb_replays"


def their_attitude(name):
    d = np.load(ROOT / f"data/gnc2/their_{name}.npz")
    R = d["R"]; q = R[int(np.argmax(np.isfinite(R[:, 0]))), 6:10]
    e0, e1, e2, e3 = q
    z = np.array([2 * (e1 * e3 + e0 * e2), 2 * (e2 * e3 - e0 * e1), 1 - 2 * (e1**2 + e2**2)])
    incl = float(np.degrees(np.arcsin(np.clip(z[2], -1, 1))))
    # match the env convention by search: the heading whose get_initial_attitude body z is closest
    best, hb = -2, 0.0
    for h in np.arange(0, 360, 0.5):
        q2 = np.array(get_initial_attitude(incl, float(h)))
        e0, e1, e2, e3 = q2
        z2 = np.array([2 * (e1 * e3 + e0 * e2), 2 * (e2 * e3 - e0 * e1), 1 - 2 * (e1**2 + e2**2)])
        if z2 @ z > best:
            best, hb = z2 @ z, float(h)
    return incl, hb


def make_cls(base, variant, name):
    class A(base):
        def decide_launch(self, obs, launch, att=0):
            super().decide_launch(obs, launch, att)
            if self.launched() and variant in ("att", "vert", "know"):
                ih = their_attitude(name) if variant in ("att", "know") else (90.0, 0.0)
                self.incl_head = np.array(ih)
                self.q = np.array(get_initial_attitude(*ih))
            if self.launched() and variant.startswith("j"):
                ih = (89.5, float(variant[1:]))
                self.incl_head = np.array(ih)
                self.q = np.array(get_initial_attitude(*ih))

        def get_action(self, obs):
            if variant == "know" and hasattr(self, "info"):
                b = int(self.info["pops"][0][1])
                if obs["balloon_status"][b, 0] == 0:
                    obs = dict(obs); st = obs["balloon_status"].copy(); bs = obs["balloon_states"].copy()
                    st[b, 0] = 1; bs[b, 2] += 2.0; bs[b, 3:] = 0.0
                    obs["balloon_status"], obs["balloon_states"] = st, bs
            return super().get_action(obs)

        def _saturate(self, a_cmd):
            if variant != "nodown":
                return super()._saturate(a_cmd)
            g = np.array([0.0, 0.0, 9.81]); a = a_cmd.copy()
            a[2] = max(a[2], -3.0 - g[2])  # vertical specific force >= -3 m/s^2
            A_, B_, C_ = a @ a, 2 * a @ g, g @ g - self.a_thrust**2
            s = 1.0
            if A_ > 1e-9 and (a + g) @ (a + g) > self.a_thrust**2:
                s = max((-B_ + np.sqrt(max(B_ * B_ - 4 * A_ * C_, 0.0))) / (2 * A_), 0.0)
            return s * a + g
    return A


def run(variant, mode, name):
    path = str(REPLAYS / f"{name}.json")
    info = rt.replay_info(path)
    params, given = rl2_env.load_scenario_parameters(1)
    env = rl2_env.BalloonPoppingEnv(render_mode=None, parameters=params)
    with contextlib.redirect_stdout(io.StringIO()):
        obs, _ = env.reset(seed=info["seed"])
    kw = dict(policy_path=CK, sl_path=CK.replace("rl3_policy", "rl2_sl"))
    if variant in ("low68", "low75"):
        kw["launch_incl"] = float(variant[3:])
    if mode == "strict":
        ag = make_cls(rt.Teacher, variant, name)(given, info, **kw)
    else:
        ag = make_cls(RL3Agent, variant, name)(given, **kw)
    T, R, S, TGT, F = [], [], [], [], []
    done = False
    while not done:
        a = ag.get_action(obs)
        obs, r, term, trunc, inf = env.step(a)
        done = term or trunc
        T.append(float(obs["simulation_time"])); R.append(np.asarray(inf["rocket_states"], float))
        S.append(obs["balloon_status"][:, 0].astype(np.int8))
        dbg = ag.debug or {}
        TGT.append(-1 if dbg.get("target") is None else int(dbg["target"]))
        F.append(np.asarray(dbg.get("f", [np.nan] * 3), float))
    S, T, R = np.array(S), np.array(T), np.array(R)
    pops = sorted((T[int(np.argmax(S[:, b] == 2))], b) for b in range(S.shape[1]) if (S[:, b] == 2).any())
    live = np.isfinite(R[:, 0]); tl0 = float(T[int(np.argmax(live))])
    np.savez_compressed(OUT / f"{variant}_{mode}_{name}.npz", t=T, R=R.astype(np.float32), TGT=np.array(TGT), F=np.array(F, np.float32),
                        pops=np.array(pops, float).reshape(-1, 2), launch=tl0, incl_head=np.asarray(ag.incl_head, float))
    p = np.array([x[0] for x in pops]) - tl0
    return variant, mode, name, len(pops), (p[0] if len(p) else np.nan), (p[-1] if len(p) else np.nan), int((p > 30).sum()), tuple(np.round(ag.incl_head, 1))


if __name__ == "__main__":
    import multiprocessing as mp
    import warnings; warnings.filterwarnings("ignore")
    variants = sys.argv[1].split(",")
    mode = sys.argv[2] if len(sys.argv) > 2 else "strict"
    names = sys.argv[3:] or "10_CoolBa 10_SpaceY 11_Mochi 11_near_miss 12_Apex 13_Apex 13_Yaka 14_SpaceBerry 16_SpaceBerry 16_TeamDJ 18_Taoyuan".split()
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(v, mode, n) for v in variants for n in names]
    with mp.get_context("spawn").Pool(4) as pool:
        res = pool.starmap(run, jobs)
    for v in variants:
        rr = [r for r in res if r[0] == v]
        for r in rr:
            print(f"{r[0]:7s} {r[1]:6s} {r[2]:15s} pops {r[3]:2d} first {r[4]:5.2f} last {r[5]:5.2f} postburn {r[6]} att {r[7]}")
        print(f"== {v} {mode}: total pops {sum(r[3] for r in rr)}  first-pop mean {np.nanmean([r[4] for r in rr]):.2f} median {np.nanmedian([r[4] for r in rr]):.2f}"
              f"  last mean {np.nanmean([r[5] for r in rr]):.2f}  postburn {sum(r[6] for r in rr)}", flush=True)

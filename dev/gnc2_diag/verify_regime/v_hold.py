"""Test of the end-of-burn fallback mechanism found in verification: with coast=1 the reach set collapses as burn_left -> 0,
RL2Agent._select_target then returns t_go = t_grid[-1] (14.9 s) and the coast guidance commands ~straight up 0.2-0.7 s before
burnout.  Variant 'hold': in the last 1.5 s of burn, if the same target just became unreachable, keep counting down the last
finite reach time instead (the command saturates toward the intercept).  Everything else unchanged.
.venv/bin/python dev/gnc2_diag/verify_regime/v_hold.py <strict|free> <base|hold> [names...]
"""
import contextlib, io, os, sys
from pathlib import Path
os.environ["SLACK"] = "inf"
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from BalloonPoppingGymEnv.agents.rl3_agent import RL3Agent  # noqa: E402
from dev import replay_teacher as rt  # noqa: E402
from dev import rl2_env  # noqa: E402

CK = str(ROOT / "runs/s4bw4/rl3_policy_25478400.npz")
OUT = Path(__file__).resolve().parent / "sim"


def mixin(base, variant):
    class H(base):
        def _select_target(self, p, v, obs):
            tgt, tg = super()._select_target(p, v, obs)
            if variant == "hold" and tgt is not None and tg >= self.t_grid[-1] - 1e-9 and getattr(self, "burn_left", 99) < 1.5:
                last = getattr(self, "_hold_last", None)
                if last is not None and last[0] == tgt and last[1] < 5.0:
                    tg = max(last[1] - self.dt, self.t_go_min)
            self._hold_last = (tgt, tg) if tgt is not None else None
            return tgt, tg
    return H


def run(mode, variant, name):
    path = str(ROOT.parent / "lb_replays" / f"{name}.json")
    info = rt.replay_info(path)
    params, given = rl2_env.load_scenario_parameters(1)
    env = rl2_env.BalloonPoppingEnv(render_mode=None, parameters=params)
    with contextlib.redirect_stdout(io.StringIO()):
        obs, _ = env.reset(seed=info["seed"])
    kw = dict(policy_path=CK, sl_path=CK.replace("rl3_policy", "rl2_sl"))
    ag = mixin(rt.Teacher, variant)(given, info, **kw) if mode == "strict" else mixin(RL3Agent, variant)(given, **kw)
    done, Tt, S, R = False, [], [], []
    while not done:
        a = ag.get_action(obs)
        obs, r, term, trunc, inf = env.step(a)
        done = term or trunc
        Tt.append(float(obs["simulation_time"])); S.append(obs["balloon_status"][:, 0].astype(np.int8)); R.append(np.asarray(inf["rocket_states"], float))
    S, Tt, R = np.array(S), np.array(Tt), np.array(R)
    pops = sorted((Tt[int(np.argmax(S[:, b] == 2))], b) for b in range(100) if (S[:, b] == 2).any())
    tl = Tt[int(np.flatnonzero(np.isfinite(R[:, 0]))[0])]
    p = np.array([x[0] for x in pops]) - tl
    np.savez_compressed(OUT / f"{variant}_{mode}_{name}.npz", pops=np.array(pops, float).reshape(-1, 2), launch=tl)
    return mode, variant, name, len(pops), int((p > 29.99).sum()), (p[p <= 29.99][-1] if (p <= 29.99).any() else np.nan)


if __name__ == "__main__":
    import multiprocessing as mp, warnings
    warnings.filterwarnings("ignore")
    mode, variant = sys.argv[1], sys.argv[2]
    names = sys.argv[3:] or "10_CoolBa 10_SpaceY 11_Mochi 11_near_miss 12_Apex 13_Apex 13_Yaka 14_SpaceBerry 16_SpaceBerry 16_TeamDJ 18_Taoyuan".split()
    OUT.mkdir(exist_ok=True)
    with mp.get_context("spawn").Pool(4) as pool:
        res = pool.starmap(run, [(mode, variant, n) for n in names])
    for r in res:
        print(f"{r[0]:6s} {r[1]:5s} {r[2]:15s} pops {r[3]:2d} post-burn(>=29.99 s) {r[4]} last pre-burn pop t+{r[5]:.2f}")
    print(f"== {mode} {variant}: total {sum(r[3] for r in res)} post-burn {sum(r[4] for r in res)}", flush=True)

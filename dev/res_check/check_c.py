"""C: a constant forced residual u (res.w = 0, res.b = u) vs the zero head on one seed: trajectory change, the commanded
force offset bound * a_thrust * tanh(u), and no residual on the pad, after burnout or in the end-of-burn mode."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import contextlib, io, logging, multiprocessing as mp, sys, warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
T = Path(__file__).parent / "tmp"
U = np.array([1.0, -0.5, 0.3])


def run(args):
    pol, seed = args
    warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
    from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv
    from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters
    from BalloonPoppingGymEnv.agents.rl3_agent import RL3Agent
    from BalloonPoppingGymEnv.agents.rl2_sl import load_sl
    params, given = load_scenario_parameters(4)
    env = BalloonPoppingEnv(render_mode=None, parameters=params)
    with contextlib.redirect_stdout(io.StringIO()):
        obs, _ = env.reset(seed=seed)
    ag = RL3Agent(given, policy_path=pol); ag.sl = load_sl(T / "rl2_sl_86630400.npz")
    rec, cur, dec = [], {}, []
    g0, s0, e0, r0 = ag._guidance, ag._saturate, ag._eob, ag.set_res
    ag._guidance = lambda *a: cur.__setitem__("g", g0(*a)) or cur["g"]
    ag._saturate = lambda a: cur.__setitem__("a_in", a.copy()) or cur.__setitem__("f_sat", s0(a)) or cur["f_sat"]
    ag._eob = lambda *a: cur.__setitem__("eob", e0(*a)) or cur["eob"]
    ag.set_res = lambda u: (r0(u), dec.append((float(obs["simulation_time"]), ag.a_thrust, np.asarray(u, float).copy(), ag.res.copy())))[0]
    done, pos = False, []
    while not done:
        cur.clear()
        t = float(obs["simulation_time"])
        a = ag.get_action(obs)
        rec.append(dict(t=t, launched=ag.launched_at is not None, burning=ag._burning(t), res=ag.res.copy(), target=ag.target,
                        eob_on=ag.eob_f is not None, f=ag.debug.get("f"), **cur))
        obs, r, te, tr, info = env.step(a); done = te or tr
        pos.append(np.asarray(info["rocket_states"][:3], float))
    return rec, dec, np.array(pos), int(info["popped_count"])


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 10002
    P = dict(np.load(T / "rl3_policy_res0.npz")); P["res.b"] = U.astype(np.float32); np.savez(T / "rl3_policy_resC.npz", **P)
    bound = float(P["res.bound"])
    with mp.Pool(2) as p:
        (rc, dc, pc, nc), (r0, d0, p0, n0) = p.map(run, [(str(T / "rl3_policy_resC.npz"), seed), (str(T / "rl3_policy_res0.npz"), seed)])
    n = min(len(pc), len(p0)); fin = np.isfinite(pc[:n]).all(1) & np.isfinite(p0[:n]).all(1)
    print(f"seed {seed}: popped forced {nc} vs zero head {n0}; steps {len(pc)} vs {len(p0)}; "
          f"max |position difference| {np.abs(pc[:n][fin] - p0[:n][fin]).max():.2f} m")
    print(f"decisions with the residual set: {len(dc)} (zero head: {len(d0)}, all res exactly 0: {all(not d[3].any() for d in d0)})")
    err = max(np.abs(d[3] - bound * d[1] * np.tanh(U)).max() for d in dc)
    print(f"|res - bound*a_thrust*tanh(u)| max {err:.2e};  |res|/a_thrust = {np.linalg.norm(dc[0][3]) / dc[0][1]:.4f} "
          f"(expected {bound * np.linalg.norm(np.tanh(U)):.4f});  first res {np.round(dc[0][3], 3)} at a_thrust {dc[0][1]:.3f}")
    fl = [r for r in rc if "g" in r]  # steps that ran the guidance (burning, a target)
    ex = sum(np.array_equal(r["a_in"], r["g"] + r["res"]) for r in fl)  # z may differ: ground avoidance raises it
    off = max(np.abs(r["a_in"] - r["g"] - r["res"])[:2].max() for r in fl)
    print(f"guidance steps {len(fl)}: saturate input == guidance + res exactly on {ex}; max |x,y difference| {off:.2e}; "
          f"min |res| on them {min(np.linalg.norm(r['res']) for r in fl):.3f}")
    pad = [r for r in rc if not r["launched"]]
    print(f"pad steps {len(pad)}: residual nonzero on {sum(r['res'].any() for r in pad)}, guidance called on {sum('g' in r for r in pad)}")
    first = next(i for i, r in enumerate(rc) if r["res"].any())
    print(f"launched, before the first decision: {sum(1 for r in rc[:first] if r['launched'])} steps with res = 0")
    post = [r for i, r in enumerate(rc) if r["launched"] and not r["burning"] and i > first]
    print(f"after burnout {len(post)} steps: guidance / saturate called on {sum('g' in r or 'a_in' in r for r in post)}")
    eob = [r for r in fl if r["eob_on"]]
    print(f"eob steps {len(eob)}: final force == eob force on {sum(np.array_equal(r['f'], r['eob']) for r in eob)}; "
          f"eob force != saturated guidance+res on {sum(not np.allclose(r['eob'], r['f_sat']) for r in eob)}")
    nt = [r for r in rc if r["burning"] and r["launched"] and r["target"] is None and "a_in" in r]
    print(f"burning steps without a target: {len(nt)}, saturate input there all zero (no residual): {all(not r['a_in'].any() for r in nt)}")

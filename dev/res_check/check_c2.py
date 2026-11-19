"""RL3_RES2 check 4: forced u[3] (stretch) or u[4] (nav_N) = 1 via res.b on the 5-output head (other rows as trained),
vs the zero rows: flight change; the stretch / N actually used in _guidance = the expected products; features (incl. SL
inputs) and T1DATA rows independent of s_mul; both multipliers 1 on the pad and before the first decision."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import contextlib, io, logging, multiprocessing as mp, sys, warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
T = Path(__file__).parent / "tmp"


def run(args):
    pol, seed = args
    warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
    from BalloonPoppingGymEnv.envs.balloon_world import BalloonPoppingEnv
    from BalloonPoppingGymEnv.evaluation.evaluate import load_scenario_parameters
    from BalloonPoppingGymEnv.agents.rl3_agent import RL3Agent
    from BalloonPoppingGymEnv.agents.rl2_sl import load_sl
    from dev.rl2_env import GuidanceEnv
    params, given = load_scenario_parameters(4)
    env = BalloonPoppingEnv(render_mode=None, parameters=params)
    with contextlib.redirect_stdout(io.StringIO()):
        obs, _ = env.reset(seed=seed)
    ag = RL3Agent(given, policy_path=pol); ag.sl = load_sl(T / "rl2_sl_99561600.npz")
    g0, d0 = ag._guidance, ag.decide
    st = dict(gdiff=0.0, used=set(), feat_same=0, feat_n=0, log_same=0)

    def guid(p, v, bp, bv, t_go):
        out = g0(p, v, bp, bv, t_go)
        s, N, sm, nm = ag.stretch, ag.nav_N, ag.s_mul, ag.n_mul  # reference: the old formula with stretch * s_mul, N * n_mul
        ag.stretch, ag.nav_N, ag.s_mul, ag.n_mul = s * sm, N * nm, 1.0, 1.0
        ref = g0(p, v, bp, bv, t_go)
        ag.stretch, ag.nav_N, ag.s_mul, ag.n_mul = s, N, sm, nm
        st["gdiff"] = max(st["gdiff"], float(np.abs(out - ref).max() / max(np.abs(ref).max(), 1e-9)))
        st["used"].add((round(s * sm, 6), round(N * nm, 6), round(sm, 6), round(nm, 6)))
        return out

    def dec(o, a):  # at each decision: features / T1DATA row with s_mul as is vs s_mul = 1
        sm = ag.s_mul
        x1 = ag.features(o); ag.s_mul = 1.0; x2 = ag.features(o); ag.s_mul = sm
        st["feat_n"] += 1; st["feat_same"] += np.array_equal(x1, x2)
        stub = type("E", (), {})(); stub.log_dir, stub.ag, stub.obs, stub.pending = "x", ag, o, []
        GuidanceEnv._log_decision(stub, x1, a); r1 = list(stub.pending); stub.pending = []
        ag.s_mul = 1.0; GuidanceEnv._log_decision(stub, x1, a); ag.s_mul = sm
        st["log_same"] += len(r1) == len(stub.pending) and all(np.array_equal(p[0], q[0]) for p, q in zip(r1, stub.pending))
        return d0(o, a)
    ag._guidance, ag.decide = guid, dec
    pad_mul, pre_dec_mul, done, pos, first = set(), set(), False, [], None
    while not done:
        t = float(obs["simulation_time"])
        a = ag.get_action(obs)
        if ag.launched_at is None or not np.isfinite(ag.launch_time) or t < ag.launch_time:
            pad_mul.add((ag.s_mul, ag.n_mul))
        elif ag.cmd is None:
            pre_dec_mul.add((ag.s_mul, ag.n_mul))
        obs, r, te, tr, info = env.step(a); done = te or tr
        pos.append(np.asarray(info["rocket_states"][:3], float))
    st["used"] = sorted(st["used"])[:6]
    return st, pad_mul, pre_dec_mul, np.array(pos), int(info["popped_count"])


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 10002
    P = dict(np.load(T / "rl3_policy_res5.npz"))
    for name, i in (("S", 3), ("N", 4)):
        Q = dict(P); Q["res.b"] = P["res.b"].copy(); Q["res.b"][i] = 1.0; np.savez(T / f"rl3_policy_res5{name}.npz", **Q)
    print(f"expected: s_mul = exp(0.3 tanh 1) = {np.exp(0.3 * np.tanh(1.0)):.6f}, n_mul = exp(0.15 tanh 1) = {np.exp(0.15 * np.tanh(1.0)):.6f}")
    with mp.Pool(3) as p:
        res = p.map(run, [(str(T / f"rl3_policy_res5{n}.npz"), seed) for n in ("", "S", "N")])
    p0 = res[0][3]
    for name, (st, pad, pre, pos, n) in zip(("zero rows", "u[3]=1", "u[4]=1"), res):
        k = min(len(pos), len(p0)); f = np.isfinite(pos[:k]).all(1) & np.isfinite(p0[:k]).all(1)
        print(f"{name}: popped {n}, steps {len(pos)}, max |pos - zero-rows pos| {np.abs(pos[:k][f] - p0[:k][f]).max():.2f} m; "
              f"guidance vs reference max rel diff {st['gdiff']:.1e}; (stretch, N, s_mul, n_mul) used e.g. {st['used'][:4]}; "
              f"features identical with s_mul toggled at {st['feat_same']}/{st['feat_n']} decisions, T1DATA rows identical at {st['log_same']}/{st['feat_n']}; "
              f"pad (s_mul, n_mul) {sorted(pad)}; launched, before the first decision {sorted(pre)}")

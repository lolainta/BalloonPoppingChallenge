"""Independent test of 'the strict Teacher is handicapped because it holds each target until it pops':
a hard, time-based handover that imitates what our free policy does (it drops a target when the GNC min reach time
to it is ~0.7 s, median, IQR 0.6-0.9, measured from ours_free TGO/TGT). No hit check, no blending: when the current
target's min reach time < TAU the GNC target becomes the next still-alive balloon of THEIR order.
Usage: vhand.py TAU TAG   -> vhand_<TAG>.json"""
import contextlib, io, json, os, sys
from pathlib import Path
os.environ["SLACK"] = "inf"
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from dev import replay_teacher as rt  # noqa: E402
from dev import rl2_env  # noqa: E402
CK = str(ROOT / "runs/s4bw4/rl3_policy_25478400.npz")
NAMES = "10_CoolBa 10_SpaceY 11_Mochi 11_near_miss 12_Apex 13_Yaka 14_SpaceBerry 16_TeamDJ 18_Taoyuan".split()  # 9 unique


class HandTeacher(rt.Teacher):
    def __init__(self, given, info, tau, **kw):
        super().__init__(given, info, **kw); self.tau = tau; self.n_hand = 0

    def _select_target(self, p, v, obs):
        b, tg = super()._select_target(p, v, obs)
        if b is None or tg is None or tg >= self.tau:
            return b, tg
        st = obs["balloon_status"][:, 0]; order = [bb for _, bb in self.info["pops"]]
        if b not in order: return b, tg
        nb = next((bb for bb in order[order.index(b) + 1:] if st[bb] == 1), None)
        if nb is None: return b, tg
        s = obs["balloon_states"][nb]
        t1 = self._reach_time(p, v, s[None, :3], s[None, 3:])[0]
        if not np.isfinite(t1): return b, tg
        self.n_hand += 1
        return nb, t1


def run(name, tau):
    import warnings; warnings.filterwarnings("ignore")
    info = rt.replay_info(str(ROOT.parent / "lb_replays" / f"{name}.json"))
    params, given = rl2_env.load_scenario_parameters(1)
    env = rl2_env.BalloonPoppingEnv(render_mode=None, parameters=params)
    with contextlib.redirect_stdout(io.StringIO()):
        obs, _ = env.reset(seed=info["seed"])
    ag = HandTeacher(given, info, tau, policy_path=CK, sl_path=CK.replace("rl3_policy", "rl2_sl"))
    done, S, T, launch = False, [], [], None
    while not done:
        obs, r, term, trunc, inf = env.step(ag.get_action(obs)); done = term or trunc
        t = float(obs["simulation_time"]); T.append(t); S.append(obs["balloon_status"][:, 0].copy())
        if launch is None and not np.isnan(np.asarray(inf["rocket_states"], float)[0]): launch = t
    S = np.array(S); T = np.array(T)
    pops = sorted((float(T[int(np.argmax(S[:, b] == 2))]), int(b)) for b in range(S.shape[1]) if (S[:, b] == 2).any())
    return dict(name=name, pops=pops, launch=launch, n_hand=ag.n_hand)


if __name__ == "__main__":
    import multiprocessing as mp
    tau, tag = float(sys.argv[1]), sys.argv[2]
    with mp.get_context("spawn").Pool(4) as p:
        res = p.starmap(run, [(n, tau) for n in NAMES])
    json.dump(res, open(Path(__file__).parent / f"vhand_{tag}.json", "w"))
    gaps = []
    for r in res:
        P = [tp for tp, _ in r["pops"] if tp <= r["launch"] + 30]; gaps += list(np.diff(P))
        print(f"{r['name']:14s} pops {len(r['pops']):2d} handover steps {r['n_hand']}")
    print(f"{tag} tau={tau}: total {sum(len(r['pops']) for r in res)} (9 unique); hop median {np.median(gaps):.2f} mean {np.mean(gaps):.2f} n={len(gaps)}")

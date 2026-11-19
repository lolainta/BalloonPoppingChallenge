"""Strict-order simulation (their launch time, their next still-alive balloon, policy speed levels) with a prototype
chaining guidance. Nothing in the repo is modified: ChainTeacher subclasses dev/replay_teacher.Teacher.

mode 0 : baseline (= ours_strict)
mode 1 : early handover  - fly the NEXT balloon's command whenever, held constant, it still passes the current balloon
                           within R_OK (else the current command)
mode 2 : hit-constrained blend - a = (1-l) a_cur + l a_next, l in 0..1, pick min [t_hit(current) + t_reach(next from the
                           predicted arrival state)] (+ turn penalty W_TURN * angle(thrust at arrival, next min-time dir) / max_rate)
Usage: chain_sim.py MODE [R_OK] [W_TURN] [TAG]   -> chain_<TAG>.json (pops / hop times per replay)
"""
import contextlib, io, json, os, sys
from pathlib import Path
os.environ["SLACK"] = "inf"
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from dev import replay_teacher as rt  # noqa: E402
from dev import rl2_env  # noqa: E402

CK = str(ROOT / "runs/s4bw4/rl3_policy_25478400.npz")
NAMES = "10_CoolBa 10_SpaceY 11_Mochi 11_near_miss 12_Apex 13_Apex 13_Yaka 14_SpaceBerry 16_SpaceBerry 16_TeamDJ 18_Taoyuan".split()
G = np.array([0.0, 0.0, -9.81])
LAM = np.linspace(0.0, 1.0, 11)
TH = np.arange(0.0, 6.0, 0.02)


class ChainTeacher(rt.Teacher):
    def __init__(self, given, info, mode=0, r_ok=1.0, w_turn=0.0, **kw):
        super().__init__(given, info, **kw)
        self.mode, self.r_ok, self.w_turn = mode, r_ok, w_turn
        self._obs = None
        self.n_chain = 0; self.n_steps = 0

    def next_after(self, obs, cur):
        st = obs["balloon_status"][:, 0]; seen = False
        for tp, b in self.info["pops"]:
            if b == cur:
                seen = True; continue
            if seen and st[b] == 1:
                return b
        return None

    def _gnc_action(self, obs):
        self._obs = obs
        return super()._gnc_action(obs)

    def _t_hit(self, p, v, net, bp, bv, horizon):
        rel = (bp - p)[None] + (bv - v)[None] * TH[:, None] - 0.5 * net[None] * TH[:, None] ** 2
        ok = (np.linalg.norm(rel, axis=1) < self.r_ok) & (TH <= horizon)
        return TH[int(np.argmax(ok))] if ok.any() else np.inf

    def _guidance(self, p, v, bpos, bvel, t_go):
        a_cur = super()._guidance(p, v, bpos, bvel, t_go)
        if self.mode == 0 or self._obs is None or self.target is None:
            return a_cur
        self.n_steps += 1
        nb = self.next_after(self._obs, self.target)
        if nb is None:
            return a_cur
        bn = self._obs["balloon_states"][nb].astype(float)
        tn = self._reach_time(p, v, bn[None, :3], bn[None, 3:])[0]
        if not np.isfinite(tn):
            return a_cur
        a_next = super()._guidance(p, v, bn[:3], bn[3:], tn)
        hor = 1.5 * max(t_go * self.stretch, self.t_go_min) + 0.5
        if self.mode == 1:
            net = self._saturate(a_next) + G
            if np.isfinite(self._t_hit(p, v, net, bpos, bvel, hor)):
                self.n_chain += 1
                return a_next
            return a_cur
        best, a_best, c0 = np.inf, a_cur, None
        for lam in LAM:
            a = (1 - lam) * a_cur + lam * a_next
            f = self._saturate(a); net = f + G
            th = self._t_hit(p, v, net, bpos, bvel, hor)
            if not np.isfinite(th):
                if lam == 0.0:
                    return a_cur  # the model says even the current command misses: leave it to the baseline
                continue
            ph, vh = p + v * th + 0.5 * net * th ** 2, v + net * th
            t2 = self._reach_time(ph, vh, (bn[:3] + bn[3:] * th)[None], bn[None, 3:])[0]
            if not np.isfinite(t2):
                continue
            c = th + t2
            if self.w_turn > 0:
                a2 = super()._guidance(ph, vh, bn[:3] + bn[3:] * th, bn[3:], t2)
                f2 = self._saturate(a2)
                ang = np.arccos(np.clip(f @ f2 / max(np.linalg.norm(f) * np.linalg.norm(f2), 1e-9), -1, 1))
                c += self.w_turn * ang / self.max_rate
            if lam == 0.0:
                c0 = c
            if c < best - 1e-9:
                best, a_best = c, a
        if c0 is not None and best < c0 - 0.02 and a_best is not a_cur:
            self.n_chain += 1
            return a_best
        return a_cur


def run(name, mode, r_ok, w_turn):
    import warnings; warnings.filterwarnings("ignore")
    path = str(ROOT.parent / "lb_replays" / f"{name}.json")
    info = rt.replay_info(path)
    params, given = rl2_env.load_scenario_parameters(1)
    env = rl2_env.BalloonPoppingEnv(render_mode=None, parameters=params)
    with contextlib.redirect_stdout(io.StringIO()):
        obs, _ = env.reset(seed=info["seed"])
    ag = ChainTeacher(given, info, mode=mode, r_ok=r_ok, w_turn=w_turn, policy_path=CK, sl_path=CK.replace("rl3_policy", "rl2_sl"))
    done, S, T, launch = False, [], [], None
    Rl = []
    while not done:
        obs, r, term, trunc, inf = env.step(ag.get_action(obs))
        done = term or trunc
        t = float(obs["simulation_time"]); T.append(t); S.append(obs["balloon_status"][:, 0].copy())
        rs = np.asarray(inf["rocket_states"], float)
        if launch is None and not np.isnan(rs[0]):
            launch = t
    S = np.array(S); T = np.array(T)
    pops = sorted((float(T[int(np.argmax(S[:, b] == 2))]), int(b)) for b in range(S.shape[1]) if (S[:, b] == 2).any())
    return dict(name=name, pops=pops, launch=launch, n_chain=ag.n_chain, n_steps=ag.n_steps)


if __name__ == "__main__":
    import multiprocessing as mp
    mode = int(sys.argv[1]); r_ok = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    w_turn = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0; tag = sys.argv[4] if len(sys.argv) > 4 else f"m{mode}"
    names = os.environ.get("NAMES", " ".join(NAMES)).split()
    with mp.get_context("spawn").Pool(4) as p:
        res = p.starmap(run, [(n, mode, r_ok, w_turn) for n in names])
    out = Path(__file__).parent / f"chain_{tag}.json"
    json.dump(res, open(out, "w"))
    tot, gaps = 0, []
    for r in res:
        P = [tp for tp, _ in r["pops"] if tp <= r["launch"] + 30]
        tot += len(r["pops"]); gaps += list(np.diff(P))
        print(f"{r['name']:14s} pops {len(r['pops']):2d}  chain steps {r['n_chain']}/{r['n_steps']}")
    print(f"{tag}: total pops {tot}; median hop {np.median(gaps):.2f} s mean {np.mean(gaps):.2f} s (n={len(gaps)})")

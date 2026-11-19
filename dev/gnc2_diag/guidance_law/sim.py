"""Strict-order flights (their launch time, always their next still-alive balloon; = dev/gnc2_log.py mode strict) with
guidance-law variants; logs per step into dev/gnc2_diag/guidance_law/sims/<variant>__<name>.npz (same keys as
data/gnc2/ours_strict_*.npz minus B velocities are kept) so hops.py-style analysis can run on them.

Variants (name:key=val,...):
  any GNC tunable (nav_N, k_att, max_rate, t_go_min, margin, ...) -> agent kwargs
  fix_s=i   : force speed level rl2_agent.STRETCH[i] (0.6, 0.8, 1.0, 1.3)
  smax=x    : cap the policy's speed level at x (e.g. 1.0: never fly slower than the min-time law)
  tol=m     : the GNC's reach time counts a hit within m of the centre (default 0 = centre); changes t_go and stops the
              terminal give-up (re-plan to a turn-around while the pass is still inside the balloon radius)
  tolc=m    : consistent version of tol: reach time with m AND aim at the nearest point of the m-sphere (edge aim)
  free=1    : our policy alone (its own launch / targets / speed levels) instead of the strict teacher
  ho=tau    : terminal handoff: when the current target's predicted closest approach (current relative velocity and
              acceleration held) is within ho_r m (default 0.7) and <= tau s ahead, retarget to the next balloon in their order;
              go back if the handed-off balloon is still alive 0.15 s after its predicted closest approach
.venv/bin/python dev/gnc2_diag/guidance_law/sim.py --procs 4 base: 's10:fix_s=2' 'ho3:ho=0.3'
"""
import argparse, contextlib, io, json, os, sys
from pathlib import Path

os.environ["SLACK"] = "inf"
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).resolve().parent / "sims"
CK = str(ROOT / "runs/s4bw4/rl3_policy_25478400.npz")
G = np.array([0.0, 0.0, -9.81])


def val(v):
    try:
        return float(v)
    except ValueError:
        return v


def make_agent(given, info, kw):
    from dev import replay_teacher as rt
    from BalloonPoppingGymEnv.agents.rl2_agent import STRETCH
    kw = dict(kw)
    rt.FIX_S = str(int(kw.pop("fix_s"))) if "fix_s" in kw else None
    smax = kw.pop("smax", None); ho = kw.pop("ho", None); ho_r = kw.pop("ho_r", 0.7); tol = kw.pop("tol", 0.0)
    tolc = kw.pop("tolc", 0.0); tol = tol or tolc

    class T(rt.Teacher):
        ho_from, ho_until = None, -1.0

        def _reach_time(self, p, v, bpos, bvel):
            """tol > 0: a hit = within tol m of the centre (the GNC's plan and its give-up test use the real radius)."""
            if not tol:
                return super()._reach_time(p, v, bpos, bvel)
            tg = self.t_grid[:, None]
            r = (bpos - p[..., None, :])[..., None, :]
            vr = (bvel - v[..., None, :])[..., None, :]
            miss = r + vr * tg - 0.5 * G * tg ** 2
            if self.bwind:
                miss = miss + self._bdrift(bpos[..., None, 2], bvel[..., None, 2], self.t_grid)
            ok = np.linalg.norm(miss, axis=-1) <= (self.a_thrust - self.margin) * self._burn_disp(self.t_grid) + tol
            return np.where(ok.any(axis=-1), self.t_grid[np.argmax(ok, axis=-1)], np.inf)

        def _guidance(self, p, v, bpos, bvel, t_go):
            """tolc > 0: consistent tolerance: the reach time counts a hit within tolc m AND the guidance aims at the nearest
            point of that sphere (the total zero-effort miss incl. gravity shrunk by tolc) -> aim at the edge, no terminal
            give-up while the pass is inside tolc."""
            if not tolc:
                return super()._guidance(p, v, bpos, bvel, t_go)
            t_go = max(t_go * self.stretch, self.t_go_min)
            if self.bwind or self.aim_prof is not None:
                bpos = bpos + self._bdrift(bpos[2], bvel[2], t_go, self.aim_prof)
            m = bpos - p + (bvel - v) * t_go - 0.5 * G * t_go ** 2
            m = m * max(1.0 - tolc / max(np.linalg.norm(m), 1e-9), 0.0)
            if self.coast and t_go > self.burn_left:
                return 0.5 * self.nav_N * m / self._burn_disp(t_go) + G
            return self.nav_N * (m + 0.5 * G * t_go ** 2) / t_go ** 2

        def pick(self, obs, t):
            st = obs["balloon_status"][:, 0]
            for tp, b in self.info["pops"]:
                if st[b] == 1 and not (b == self.ho_from and t < self.ho_until):
                    return b
            return None

        def needs_decision(self, obs):
            if super().needs_decision(obs):
                return True
            if ho is None or self.cmd is None:
                return False
            t = float(obs["simulation_time"]); st = obs["balloon_status"][:, 0]
            if self.ho_from is not None and st[self.ho_from] == 1 and t >= self.ho_until and self.cmd[0] != self.ho_from:
                self.ho_from = None  # missed: go back to it
                return True
            b = self.cmd[0]
            if b == self.ho_from or st[b] != 1:
                return False
            bs = obs["balloon_states"][b]
            f = self.debug.get("f") if self.debug else None
            acc = (np.asarray(f) + G) if f is not None else G
            r, vr, ar = bs[:3] - self.p_hat, bs[3:] - self.v_hat, -acc
            tau = np.arange(0.0, ho + 0.3, 0.01)
            d = np.linalg.norm(r[None] + vr[None] * tau[:, None] + 0.5 * ar[None] * tau[:, None] ** 2, axis=1)
            i = int(np.argmin(d))
            if tau[i] <= ho and d[i] <= ho_r and i < len(tau) - 1:
                self.ho_from, self.ho_until = b, t + tau[i] + 0.15
                return True
            return False

        def decide(self, obs, action):
            super().decide(obs, action)
            if smax is not None and self.cmd is not None:
                self.cmd = (self.cmd[0], min(self.cmd[1], smax))

    return T(given, info, policy_path=CK, sl_path=CK.replace("rl3_policy", "rl2_sl"), **kw)


def make_free_agent(given, kw):
    """Our policy alone (free mode) + the same GNC variants; ho: at a handoff the policy re-decides with the handed-off
    balloon masked out (it is still counted if the rocket coasts through it)."""
    from BalloonPoppingGymEnv.agents.rl3_agent import RL3Agent
    kw = dict(kw)
    ho = kw.pop("ho", None); ho_r = kw.pop("ho_r", 0.7); tol = kw.pop("tol", 0.0); smax = kw.pop("smax", None)
    tolc = kw.pop("tolc", 0.0); tol = tol or tolc
    assert "fix_s" not in kw

    class A(RL3Agent):
        ho_from, ho_until = None, -1.0

        def _reach_time(self, p, v, bpos, bvel):
            if not tol:
                return super()._reach_time(p, v, bpos, bvel)
            tg = self.t_grid[:, None]
            r = (bpos - p[..., None, :])[..., None, :]
            vr = (bvel - v[..., None, :])[..., None, :]
            miss = r + vr * tg - 0.5 * G * tg ** 2
            if self.bwind:
                miss = miss + self._bdrift(bpos[..., None, 2], bvel[..., None, 2], self.t_grid)
            ok = np.linalg.norm(miss, axis=-1) <= (self.a_thrust - self.margin) * self._burn_disp(self.t_grid) + tol
            return np.where(ok.any(axis=-1), self.t_grid[np.argmax(ok, axis=-1)], np.inf)

        def _guidance(self, p, v, bpos, bvel, t_go):
            """tolc > 0: consistent tolerance: the reach time counts a hit within tolc m AND the guidance aims at the nearest
            point of that sphere (the total zero-effort miss incl. gravity shrunk by tolc) -> aim at the edge, no terminal
            give-up while the pass is inside tolc."""
            if not tolc:
                return super()._guidance(p, v, bpos, bvel, t_go)
            t_go = max(t_go * self.stretch, self.t_go_min)
            if self.bwind or self.aim_prof is not None:
                bpos = bpos + self._bdrift(bpos[2], bvel[2], t_go, self.aim_prof)
            m = bpos - p + (bvel - v) * t_go - 0.5 * G * t_go ** 2
            m = m * max(1.0 - tolc / max(np.linalg.norm(m), 1e-9), 0.0)
            if self.coast and t_go > self.burn_left:
                return 0.5 * self.nav_N * m / self._burn_disp(t_go) + G
            return self.nav_N * (m + 0.5 * G * t_go ** 2) / t_go ** 2

        def forward(self, x):
            out = list(super().forward(x))
            if self.ho_from is not None and self._t_now < self.ho_until:
                out[0] = out[0].copy(); out[0][self.ho_from] = -1e9
            return tuple(out)

        def get_action(self, obs):
            self._t_now = float(obs["simulation_time"])
            return super().get_action(obs)

        def needs_decision(self, obs):
            if super().needs_decision(obs):
                return True
            if ho is None or self.cmd is None:
                return False
            t = float(obs["simulation_time"]); st = obs["balloon_status"][:, 0]
            b = self.cmd[0]
            if b == self.ho_from or st[b] != 1:
                return False
            bs = obs["balloon_states"][b]
            f = self.debug.get("f") if self.debug else None
            acc = (np.asarray(f) + G) if f is not None else G
            r, vr, ar = bs[:3] - self.p_hat, bs[3:] - self.v_hat, -acc
            tau = np.arange(0.0, ho + 0.3, 0.01)
            d = np.linalg.norm(r[None] + vr[None] * tau[:, None] + 0.5 * ar[None] * tau[:, None] ** 2, axis=1)
            i = int(np.argmin(d))
            if tau[i] <= ho and d[i] <= ho_r and i < len(tau) - 1 and (st == 1).sum() > 1:
                self.ho_from, self.ho_until = b, t + tau[i] + 0.15
                return True
            return False

        def decide(self, obs, action):
            super().decide(obs, action)
            if smax is not None and self.cmd is not None:
                self.cmd = (self.cmd[0], min(self.cmd[1], smax))

    return A(given, policy_path=CK, sl_path=CK.replace("rl3_policy", "rl2_sl"), **kw)


def run(vname, info, kw, scenario=1):
    import warnings; warnings.filterwarnings("ignore")
    from dev import rl2_env
    params, given = rl2_env.load_scenario_parameters(scenario)
    env = rl2_env.BalloonPoppingEnv(render_mode=None, parameters=params)
    with contextlib.redirect_stdout(io.StringIO()):
        obs, _ = env.reset(seed=info["seed"])
    free = kw.get("free", 0)
    kw = {k: v for k, v in kw.items() if k != "free"}
    ag = make_free_agent(given, kw) if free else make_agent(given, info, kw)
    sel = ag._select_target; rec = {"tgo": np.nan}
    def select(p, v, o):
        b, tg = sel(p, v, o); rec["tgo"] = np.nan if tg is None else tg; return b, tg
    ag._select_target = select
    L = {k: [] for k in "t R B S TGT TGO STRETCH F".split()}
    done = False
    while not done:
        a = ag.get_action(obs)
        obs, r, term, trunc, inf = env.step(a)
        done = term or trunc
        dbg = ag.debug or {}
        L["t"].append(float(obs["simulation_time"])); L["R"].append(np.asarray(inf["rocket_states"], float))
        L["B"].append(obs["balloon_states"].astype(np.float32)); L["S"].append(obs["balloon_status"][:, 0].astype(np.int8))
        L["TGT"].append(-1 if dbg.get("target") is None else int(dbg["target"])); L["TGO"].append(rec["tgo"])
        L["STRETCH"].append(float(ag.stretch)); L["F"].append(np.asarray(dbg.get("f", [np.nan] * 3), float))
    S = np.array(L["S"]); t = np.array(L["t"])
    pops = sorted((t[int(np.argmax(S[:, b] == 2))], b) for b in range(S.shape[1]) if (S[:, b] == 2).any())
    R = np.array(L["R"]); live = ~np.isnan(R[:, 0])
    np.savez_compressed(OUT / f"{vname}__{info['name']}.npz", **{k: np.array(v) for k, v in L.items()},
                        launch=float(t[int(np.argmax(live))]), pops=np.array(pops, float), seed=info["seed"])
    return [p[0] for p in pops]


if __name__ == "__main__":
    import multiprocessing as mp
    ap = argparse.ArgumentParser()
    ap.add_argument("variants", nargs="+"); ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--only", default="")
    ap.add_argument("--scenario", type=int, default=1)
    ap.add_argument("--seeds", default="", help="a:b -> free-mode runs on these seeds instead of the replays (e.g. scenario 4)")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    infos = json.loads((ROOT / "data/gnc2/replay_info.json").read_text())
    if a.only:
        infos = [i for i in infos if i["name"] in a.only.split(",")]
    if a.seeds:
        lo, hi = map(int, a.seeds.split(":"))
        infos = [dict(name=f"s{a.scenario}_{sd}", seed=sd, launch=0.0, score=0, pops=[]) for sd in range(lo, hi)]
    var = []
    for s in a.variants:
        name, _, rest = s.partition(":")
        var.append((name, {k: val(v) for k, v in (x.split("=", 1) for x in rest.split(",") if x)}))
    tasks = [(name, info, kw, a.scenario) for name, kw in var for info in infos]
    with mp.get_context("spawn").Pool(min(a.procs, 4, len(tasks))) as p:
        res = p.starmap(run, tasks)
    their = sum(i["score"] for i in infos) or 1
    for vi, (name, kw) in enumerate(var):
        r = res[vi * len(infos):(vi + 1) * len(infos)]
        gaps = [g for pt in r for g in np.diff(pt)]
        first = [pt[0] - i["launch"] for pt, i in zip(r, infos) if pt]
        tot = sum(len(pt) for pt in r)
        print(f"{name:10} pops {tot:3} / {their}  gap median {np.median(gaps) if gaps else float('nan'):.2f}s mean {np.mean(gaps) if gaps else float('nan'):.2f}s  "
              f"first {np.median(first) if first else float('nan'):.1f}s  per replay {[len(pt) for pt in r]}  {kw}", flush=True)

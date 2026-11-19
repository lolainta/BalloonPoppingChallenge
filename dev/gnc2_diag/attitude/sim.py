"""Strict-order counterfactual flights (their launch time, their pop order) with different ATTITUDE controllers.

Same as dev/gnc2_log.py ours(..., "strict") but: GNC keyword overrides, optional attitude-law override (subclass,
no existing file is modified), optional physics counterfactual (rocket inertia scaled -> faster rotation), and a
compact log -> dev/gnc2_diag/attitude/runs/<variant>/<name>.npz

.venv/bin/python dev/gnc2_diag/attitude/sim.py <variant> [<variant> ...]      (<= 4 processes)
"""
import contextlib, copy, io, json, os, sys, time
from pathlib import Path

os.environ["SLACK"] = "inf"
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from dev import replay_teacher as rt  # noqa: E402
from dev import rl2_env  # noqa: E402
from BalloonPoppingGymEnv.agents.gnc_agent import quat_to_R, unit  # noqa: E402

CK = str(ROOT / "runs/s4bw4/rl3_policy_25478400.npz")
REPLAYS = ROOT.parent / "lb_replays"
NAMES = "10_CoolBa 10_SpaceY 11_Mochi 11_near_miss 12_Apex 13_Apex 13_Yaka 14_SpaceBerry 16_SpaceBerry 16_TeamDJ 18_Taoyuan".split()
OUT = Path(__file__).resolve().parent / "runs"

# variant -> (GNC kwargs, attitude-law options, inertia scale)
VARIANTS = {
    "base": ({}, {}, 1.0),
    # gains only
    "katt5_mr4_kr15": (dict(k_att=5.0, max_rate=4.0, k_rate=15.0), {}, 1.0),
    "katt4_mr3": (dict(k_att=4.0, max_rate=3.0), {}, 1.0),
    "mr4": (dict(max_rate=4.0), {}, 1.0),
    "katt2_mr1.5": (dict(k_att=2.0, max_rate=1.5), {}, 1.0),  # slower (older GNC) -> sensitivity in the other direction
    # structure: time-optimal-like outer loop (sqrt law) + per-axis (box) gimbal limit + guidance-rate feed-forward
    "sqrt": ({}, dict(law="sqrt", a_dec=4.0, k_lin=8.0, max_rate=4.0, k_rate=12.0), 1.0),
    "sqrt_box_ff": ({}, dict(law="sqrt", a_dec=4.0, k_lin=8.0, max_rate=4.0, k_rate=12.0, box=True, ff=True), 1.0),
    # conservative sqrt law (outer gain below the inner-loop bandwidth ~5.4 1/s), per-axis box limit alone, ff alone
    "sqrt_c": ({}, dict(law="sqrt", a_dec=2.5, k_lin=4.0, max_rate=3.5, k_rate=10.0), 1.0),
    "box": ({}, dict(box=True), 1.0),
    "ff": ({}, dict(ff=True), 1.0),
    # end-game hold: when the target's min reach time < hold s, hold attitude (w_cmd = 0) instead of chasing F
    "hold03": ({}, dict(hold=0.3), 1.0),
    "hold05": ({}, dict(hold=0.5), 1.0),
    # physics counterfactual: inertia / 4 -> 4x angular acceleration (upper bound of what faster pointing can buy)
    "fast_physics": (dict(k_att=5.0, max_rate=5.0, k_rate=4.0), {}, 0.25),
    "fast_physics8": (dict(k_att=6.0, max_rate=8.0, k_rate=3.0), {}, 1 / 8),
}


def make_agent_cls(opt):
    base = rt.Teacher
    if not opt:
        return base

    class Att(base):
        def _select_target(self, p, v, obs):
            b, tg = super()._select_target(p, v, obs)
            self._t1 = np.inf if tg is None else float(tg)  # min reach time of the target at full thrust
            return b, tg

        def _attitude_control(self, f_des, gyro):
            R = quat_to_R(self.q)
            if opt.get("hold") and getattr(self, "_t1", np.inf) < opt["hold"]:  # end-game: stop chasing F, damp rates
                tvc = self.k_rate * (0.0 - gyro[:2])
                tvc *= min(1.0, self.max_gimbal / max(np.linalg.norm(tvc), 1e-9))
                return tvc, np.clip(-self.k_roll * gyro[2], -self.max_roll, self.max_roll)
            d_body = R.T @ unit(f_des)
            err = np.cross([0.0, 0.0, 1.0], d_body)
            ang = np.arctan2(np.linalg.norm(err), d_body[2])
            if opt.get("law") == "sqrt":  # decelerate at a_dec: |w| = sqrt(2 a_dec ang), linear k_lin ang near zero
                mag = min(opt["max_rate"], np.sqrt(2 * opt["a_dec"] * ang), opt["k_lin"] * ang)
            else:
                mag = min(opt.get("max_rate", self.max_rate), self.k_att * ang)
            w_cmd = mag * unit(err)
            if opt.get("ff"):  # rotation rate of the wanted direction itself (body frame), feed-forward
                dw = getattr(self, "_d_prev", None)
                dn = unit(f_des)
                if dw is not None:
                    w_ff = R.T @ (np.cross(dw, dn) / self.dt)
                    n = np.linalg.norm(w_ff)
                    w_cmd = w_cmd + w_ff * min(1.0, 2.0 / max(n, 1e-9))
                self._d_prev = dn
            k_rate = opt.get("k_rate", self.k_rate)
            tvc = k_rate * (w_cmd[:2] - gyro[:2])
            if opt.get("box"):
                tvc = np.clip(tvc, -self.max_gimbal, self.max_gimbal)
            else:
                tvc *= min(1.0, self.max_gimbal / max(np.linalg.norm(tvc), 1e-9))
            roll = np.clip(-self.k_roll * gyro[2], -self.max_roll, self.max_roll)
            return tvc, roll

    return Att


def run(variant, name):
    kw, opt, iscale = VARIANTS[variant]
    info = rt.replay_info(str(REPLAYS / f"{name}.json"))
    params, given = rl2_env.load_scenario_parameters(1)
    if iscale != 1.0:
        params = copy.deepcopy(params)
        rb = params["rocket"]["rocket_body"]
        rb["inertia"] = [rb["inertia"][0] * iscale, rb["inertia"][1] * iscale, rb["inertia"][2]]  # roll axis unchanged
        mo = params["rocket"]["motor"]
        if "dry_inertia" in mo:
            mo["dry_inertia"] = [mo["dry_inertia"][0] * iscale, mo["dry_inertia"][1] * iscale, mo["dry_inertia"][2]]
    env = rl2_env.BalloonPoppingEnv(render_mode=None, parameters=params)
    with contextlib.redirect_stdout(io.StringIO()):
        obs, _ = env.reset(seed=info["seed"])
    ag = make_agent_cls(opt)(given, info, policy_path=CK, sl_path=CK.replace("rl3_policy", "rl2_sl"), **kw)
    L = {k: [] for k in "t R S A F TGT STRETCH".split()}
    done, t0 = False, time.time()
    while not done:
        a = ag.get_action(obs)
        obs, r, term, trunc, inf = env.step(a)
        done = term or trunc
        dbg = ag.debug or {}
        L["t"].append(float(obs["simulation_time"])); L["R"].append(np.asarray(inf["rocket_states"], float))
        L["S"].append(obs["balloon_status"][:, 0].astype(np.int8))
        L["A"].append([*np.asarray(a["tvc"], float), float(a["roll"]), float(a["throttle"])])
        L["F"].append(np.asarray(dbg.get("f", [np.nan] * 3), float)); L["TGT"].append(-1 if dbg.get("target") is None else int(dbg["target"]))
        L["STRETCH"].append(float(ag.stretch))
    S = np.array(L["S"]); t = np.array(L["t"])
    pops = sorted((t[int(np.argmax(S[:, b] == 2))], b) for b in range(S.shape[1]) if (S[:, b] == 2).any())
    R = np.array(L["R"]); live = ~np.isnan(R[:, 0])
    out = OUT / variant; out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"{name}.npz", t=t, R=R, S=S, A=np.array(L["A"]), F=np.array(L["F"]), TGT=np.array(L["TGT"]),
                        STRETCH=np.array(L["STRETCH"]), launch=float(t[int(np.argmax(live))]), pops=np.array(pops, float))
    launch = float(t[int(np.argmax(live))])
    pt = [p for p, _ in pops if p <= launch + 30.0]
    hop = np.diff([launch] + pt)
    return f"{variant:<16} {name:<14} pops {len(pops):2d} (burn {len(pt):2d})  median hop {np.median(hop[1:]) if len(hop) > 1 else np.nan:5.2f}s  first {hop[0] if len(hop) else np.nan:5.2f}s  wall {time.time() - t0:5.0f}s"


if __name__ == "__main__":
    import multiprocessing as mp
    import warnings; warnings.filterwarnings("ignore")
    variants = sys.argv[1:]
    names = os.environ.get("NAMES", " ".join(NAMES)).split()
    with mp.get_context("spawn").Pool(int(os.environ.get("NPROC", 4))) as p:
        res = [p.apply_async(run, (v, n)) for v in variants for n in names]
        for r in res:
            print(r.get(), flush=True)

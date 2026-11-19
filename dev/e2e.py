"""Hybrid E2E (DECISION_LOG §249): after launch a recurrent policy commands the thrust every K sim steps (0.08 s); the GNC keeps
its navigation (with att_fix) and attitude loop (gnc_agent.f_ext), RL3Agent (a frozen checkpoint) still decides the launch.
  record:  fly the current agent (RL3 + GNC) and save (obs, GNC thrust / a_thrust) every K steps -> imitation data
  fly:     fly an E2E net (mean or sampled actions)
  record with E2E_NET=NET.pt: DAgger -- the net flies, the teacher (RL3 + GNC) runs in the shadow and labels every decision
Action (§259 pointer head): pick one of the M kept balloons and fly its token's GNC command plus a residual.
Observation = RL3 global features + the M = 32 balloon tokens with the shortest reach time (current-target flag zeroed:
there is no target) + body z axis, throttle, last u; stored as float16.
Action u in R^3 -> thrust demand f = a_thrust * u / max(1, |u|) (world frame).
   .venv/bin/python dev/e2e.py record OUT_DIR SEED [SEED ...]    (E2E_KW="policy_path=... sl_path=... att_fix=0.07")
   .venv/bin/python dev/e2e.py fly NET.pt SEED                    (NET.pt: torch state dict of E2ENet, or 'random')"""
import contextlib, io, json, logging, os, sys, warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from BalloonPoppingGymEnv.agents.gnc_agent import GNCAgent, quat_to_R
from BalloonPoppingGymEnv.agents.rl2_agent import F, N_MAX
from BalloonPoppingGymEnv.agents.rl3_agent import G_DIM, RL3Agent

K = int(os.environ.get("E2E_K", 8))  # sim steps per E2E decision (0.08 s; finer than the RL3 fast head)
M = 32  # balloon tokens kept: the M with the shortest GNC reach time (the rest only in the global radar features)
EXTRA = 7  # body z (3), throttle, last u (3)
FE = F + 3  # token: the RL3 token features + the thrust the GNC would command to chase that balloon now (§257)
NT = G_DIM + M * FE  # end of the tokens in the E2E observation
D = NT + EXTRA  # 647 (0.5 MB per episode as float16, what the async workers can ship)
COMMIT = int(os.environ.get("E2E_COMMIT", 0))  # §261 probe: decisions a net's new target pick is held (unless it dies / leaves the kept)
HAND = ("nearest", "hold")  # hand-coded E2E controllers for control-path checks (net = the name)
HOURS = [int(h) for h in os.environ.get("E2E_HOURS", "0:3:6:9:12:15:18:21").split(":")]


class E2EAgent(RL3Agent):
    """RL3Agent until launch; after it, e2e=True flies the GNC with the thrust demand in self.f_ext (set by the caller)."""

    def __init__(self, given, e2e=False, **kw):
        super().__init__(given, **kw)
        self.e2e, self.f_ext, self.u_prev = e2e, None, np.zeros(3)

    def flying(self, obs):
        return self.p_hat is not None and self._burning(float(obs["simulation_time"]))

    def e2e_obs(self, obs, flag=False):
        x = self.features(obs)
        tok = x[G_DIM:].reshape(N_MAX, F).copy()
        if not flag:  # "current target" flag: none in E2E (the imitation data must not use it); flag: the residual mode's teacher's
            tok[:, 8] = 0.0
        alive = tok[:, 14] > 0.5
        order = self.order = np.argsort(np.where(alive, tok[:, 6], np.inf))[:M]  # the M nearest by reach time (col 6), dead ones last
        keep = np.zeros((M, FE), np.float32)  # (dead padding rows stay all zero)
        keep[:, :F] = tok[order] * alive[order, None]
        # §257: imitating the thrust from the raw geometry flew 0.04 pops (89% of the variance explained is not intercept
        # precision) -> each kept balloon also carries the GNC's own command to chase it now (ZEM at stretch 1, t_go = its
        # reach time, wind drift, saturated with gravity compensation) in units of the thrust acceleration: the policy picks /
        # adjusts instead of re-deriving guidance
        saved = (self.stretch, self.s_mul, self.n_mul)
        self.stretch = self.s_mul = self.n_mul = 1.0
        bs = obs["balloon_states"]
        for j, b in enumerate(order):
            if alive[b]:
                t_go = max(float(tok[b, 6]) * 10.0, self.t_go_min)
                keep[j, F:] = self._saturate(self._guidance(self.p_hat, self.v_hat, bs[b, :3], bs[b, 3:], t_go)) / self.a_thrust
        self.stretch, self.s_mul, self.n_mul = saved
        return np.concatenate([x[:G_DIM], keep.ravel(), quat_to_R(self.q)[:, 2], [self.throttle], self.u_prev]).astype(np.float32)

    def set_u(self, u):
        u = np.asarray(u, float); self.u_prev = u
        self.f_ext = self.a_thrust * u / max(1.0, float(np.linalg.norm(u)))

    def get_action(self, obs):
        if self.e2e and self.p_hat is not None:
            return GNCAgent.get_action(self, obs)  # skip the RL3 / RL2 decisions: the E2E thrust drives the GNC
        return super().get_action(obs)


def make_net(H=128):
    import torch
    from torch import nn

    class E2ENet(nn.Module):
        def __init__(s):
            super().__init__()
            s.g = nn.Sequential(nn.Linear(G_DIM + EXTRA, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh())
            s.t = nn.Sequential(nn.Linear(FE, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh())
            s.q = nn.Linear(128, 64)
            s.inp = nn.Linear(192, H)
            s.gru = nn.GRU(H, H, batch_first=True)
            s.q2 = nn.Linear(H, 64)  # pointer query from the GRU state
            s.mu, s.v = nn.Linear(H, 3), nn.Linear(H, 1)  # residual on the picked command, value
            s.logstd = nn.Parameter(torch.full((3,), -2.0))
            nn.init.zeros_(s.mu.bias); s.mu.weight.data.mul_(0.1)

        def encode(s, x):  # (..., D) -> GRU input (..., H), token embeddings (..., M, 64), alive mask (..., M)
            g = torch.cat([x[..., :G_DIM], x[..., NT:]], -1)
            tok = x[..., G_DIM:NT].reshape(*x.shape[:-1], M, FE)
            mask = tok[..., 14] > 0.5
            ge, te = s.g(g), s.t(tok)
            sc = (te @ s.q(ge).unsqueeze(-1)).squeeze(-1) / 8.0
            w = torch.softmax(sc.masked_fill(~mask, -1e9), -1) * mask.any(-1, keepdim=True)
            return torch.tanh(s.inp(torch.cat([ge, (w.unsqueeze(-1) * te).sum(-2)], -1))), te, mask

        def forward(s, x, h0=None):  # x (B, T, D), h0 (1, B, H) -> pointer logits (B, T, M), residual (B, T, 3), v (B, T), h
            z, te, mask = s.encode(x)
            out, h = s.gru(z, h0)
            lg = (te @ s.q2(out).unsqueeze(-1)).squeeze(-1) / 8.0
            return lg.masked_fill(~mask, -1e9), s.mu(out), s.v(out).squeeze(-1), h

    return E2ENet()


def cmds(x):  # (..., D) -> the kept balloons' GNC commands (..., M, 3) (numpy or torch)
    return x[..., G_DIM:NT].reshape(*x.shape[:-1], M, FE)[..., F:]


def load(path):  # a hand controller's name, or an E2ENet state dict
    if path in HAND:
        return path
    import torch
    net = make_net(); net.load_state_dict(torch.load(path, map_location="cpu")); net.eval()
    return net


RES_B = float(os.environ.get("E2E_RES_B", 0.5))  # residual mode: correction = RES_B * a_thrust * tanh(d)


def fly(seed, hour=None, net=None, record=False, sample=False, kw=None, rng=None, res=False):
    """One scenario-40 episode -> dict(X (T, D), Y (T, 3) GNC thrust / a_thrust (record), P (T,) the teacher's target's slot
    in the kept tokens (record; -1: none / not kept), U (T, 3) actions, I (T,) the net's picks, R (T,) pops per decision interval
    (post-burn pops on the last), pops, launch, seed, hour; res: LP (T,) log-probabilities, V (T,) values).
    res=True (§262): the teacher (RL3 + GNC) flies as usual and the net adds a thrust correction (tanh-bounded) every K steps."""
    from dev import rl2_env
    from dev.run_seeds import parse_kw
    params, given = rl2_env.load_scenario_parameters(40)
    hour = int(rng.choice(HOURS)) if hour is None and rng is not None else (21 if hour is None else int(hour))
    params["environment"]["date"] = list(params["environment"]["date"][:3]) + [hour]
    env = rl2_env.BalloonPoppingEnv(render_mode=None, parameters=params)
    obs, _ = rl2_env.reset_clean(env, seed)
    kw = kw if kw is not None else parse_kw(os.environ.get("E2E_KW", "").split())
    ag = E2EAgent(given, e2e=net is not None and net != "hold" and not record and not res, **kw)  # (record + net: DAgger)
    X, Y, P, U, R, I, LP, V = [], [], [], [], [], [], [], []
    h, k, done, pops, tgt, held = None, 0, False, 0, None, 0
    if net is not None:
        import torch
    while not done:
        dec = ag.flying(obs) and k % K == 0
        if ag.flying(obs):
            k += 1
        if dec:
            x = ag.e2e_obs(obs, flag=res); X.append(x); R.append(0.0)
            if res:
                with torch.no_grad():
                    _, mu, v, h = net(torch.from_numpy(x).float()[None, None], h)
                mu, ls = mu[0, 0].numpy().astype(float), net.logstd.detach().numpy().astype(float)
                d = mu + np.exp(ls) * (rng or np.random).standard_normal(3) if sample else mu
                LP.append(float((-0.5 * ((d - mu) / np.exp(ls)) ** 2 - ls - 0.5 * np.log(2 * np.pi)).sum()))
                V.append(float(v[0, 0])); U.append(d)
                ag.u_prev = d; ag.f_res = RES_B * ag.a_thrust * np.tanh(d)
            elif net == "hold":  # §261: the teacher flies, its thrust sampled every K steps and held (the cost of the hold)
                ag.f_ext = None
            elif net == "nearest":  # control-path check (§259): chase the shortest-reach balloon with its token command
                u = cmds(x)[0].astype(float); ag.set_u(u); U.append(u); I.append(0)
            elif net is not None:
                with torch.no_grad():
                    lg, res, _, h = net(torch.from_numpy(x).float()[None, None], h)
                lg, res = lg[0, 0].numpy().astype(float), res[0, 0].numpy().astype(float)
                if sample:
                    pr = np.exp(lg - lg.max()); i = (rng or np.random).choice(M, p=pr / pr.sum())
                    res = res + np.exp(net.logstd.detach().numpy()) * (rng or np.random).standard_normal(3)
                else:
                    i = int(lg.argmax())
                if COMMIT and tgt is not None and held < COMMIT and obs["balloon_status"][tgt, 0] == 1 and (ag.order[:M] == tgt).any():
                    i = int(np.flatnonzero(ag.order == tgt)[0])
                if ag.order[i] != tgt:
                    tgt, held = int(ag.order[i]), 0
                held += 1
                u = cmds(x)[i].astype(float) + res; ag.set_u(u); U.append(u); I.append(i)
        act = ag.get_action(obs)
        if dec and net == "hold":
            ag.set_u(np.asarray(ag.f_gnc, float) / ag.a_thrust)
        if dec and record:  # the label: this step's GNC thrust demand (in units of the thrust acceleration)
            y = np.asarray(ag.f_gnc, float) / ag.a_thrust
            j = np.flatnonzero(ag.order == ag.target) if ag.target is not None else []
            Y.append(y); P.append(int(j[0]) if len(j) else -1)
            if net is None:
                ag.u_prev = y
        k0 = int((obs["balloon_status"][:, 0] == 2).sum())
        obs, r, te, tr, info = env.step(act)
        done = te or tr
        n = int((obs["balloon_status"][:, 0] == 2).sum())
        if n > k0 and R:
            R[-1] += n - k0
        pops = n
    f32 = lambda a, w: np.array(a, np.float32).reshape(-1, w)  # noqa: E731
    return dict(X=f32(X, D).astype(np.float16), Y=f32(Y, 3), P=np.array(P, np.int16), U=f32(U, 3), I=np.array(I, np.int16),
                LP=np.array(LP, np.float32), V=np.array(V, np.float32), R=np.array(R, np.float32), pops=pops,
                launch=float(ag.launched_at or -1), seed=int(seed), hour=hour)


def _record_one(args):
    out, seed, hour, net_path = args
    warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
    try:
        import torch; torch.set_num_threads(1)
        with contextlib.redirect_stdout(io.StringIO()):
            ep = fly(seed, hour, net=load(net_path) if net_path else None, record=True)
    except Exception as e:
        return f"seed {seed} failed: {e!r}"
    tmp = Path(out) / f".{seed}.tmp.npz"; np.savez(tmp, **ep); tmp.replace(Path(out) / f"{seed}.npz")
    return json.dumps(dict(seed=seed, hour=hour, pops=ep["pops"], steps=len(ep["X"]), launch=round(ep["launch"], 2)))


GAMMA = float(os.environ.get("E2E_GAMMA", 0.998))  # per decision (0.08 s): about 0.975 per second


def returns(R, gamma=GAMMA):
    G, g = np.zeros_like(R), 0.0
    for t in range(len(R) - 1, -1, -1):
        g = R[t] + gamma * g; G[t] = g
    return G


def bc(demo_dir, out, epochs=40, lr=1e-3, bs=32):
    """Imitation start over whole episodes (masked padding), 10% of the episodes held out: the pointer learns the teacher's
    pick (its target's slot P when recorded and kept, else the kept balloon whose command is closest to the teacher's
    thrust; cross entropy), the residual the rest (MSE),
    the critic the discounted pops. demo_dir: directories separated by ':' (DAgger rounds)."""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu":  # (a CPU node: all the job's cores)
        torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    eps = [dict(np.load(f)) for d in demo_dir.split(":") for f in sorted(Path(d).glob("[0-9]*.npz"))]
    eps = [e for e in eps if len(e["X"]) > 10]
    rng = np.random.default_rng(0); idx = rng.permutation(len(eps)); nv = max(1, len(eps) // 10)
    val, tr = [eps[i] for i in idx[:nv]], [eps[i] for i in idx[nv:]]
    T = max(len(e["X"]) for e in eps)

    def batch(es):
        X = np.zeros((len(es), T, D), np.float32); Y = np.zeros((len(es), T, 3), np.float32)
        G = np.zeros((len(es), T), np.float32); Mk = np.zeros((len(es), T), np.float32); Pt = np.full((len(es), T), -1)
        for i, e in enumerate(es):
            n = len(e["X"]); X[i, :n] = e["X"]; Y[i, :n] = e["Y"]; G[i, :n] = returns(e["R"]); Mk[i, :n] = 1
            if "P" in e:
                Pt[i, :n] = e["P"]
        C = cmds(X); alive = X[..., G_DIM:NT].reshape(len(es), T, M, FE)[..., 14] > 0.5
        I = np.where(alive, ((C - Y[..., None, :]) ** 2).sum(-1), np.inf).argmin(-1)  # the teacher's pick (proxy)
        I = np.where(Pt >= 0, Pt, I)
        Rs = Y - np.take_along_axis(C, I[..., None, None], -2)[..., 0, :]  # (no alive balloon: pick 0 = zeros, residual = Y)
        Mp = Mk * alive.any(-1)
        return [torch.from_numpy(a).to(dev) for a in (X, Y, G, Mk, I, Rs, Mp.astype(np.float32))]

    net = make_net().to(dev); opt = torch.optim.Adam(net.parameters(), lr=lr)
    Vb = batch(val); yvar = float(((Vb[1] - (Vb[1] * Vb[3][..., None]).sum((0, 1)) / Vb[3].sum()) ** 2 * Vb[3][..., None]).sum() / Vb[3].sum())

    def loss(B):
        X, Y, G, Mk, I, Rs, Mp = B
        lg, res, v, _ = net(X)
        lp = (torch.nn.functional.cross_entropy(lg.flatten(0, 1), I.flatten(), reduction="none").view_as(Mp) * Mp).sum() / Mp.sum()
        lr_ = (((res - Rs) ** 2).sum(-1) * Mk).sum() / Mk.sum()
        lv = (((v - G) ** 2) * Mk).sum() / Mk.sum()
        u = torch.gather(cmds(X), -2, lg.argmax(-1)[..., None, None].expand(-1, -1, 1, 3))[..., 0, :] + res  # (mean action)
        la = (((u - Y) ** 2).sum(-1) * Mk).sum() / Mk.sum()
        acc = (((lg.argmax(-1) == I).float()) * Mp).sum() / Mp.sum()
        return lp, lr_, lv, la, acc

    for ep in range(epochs):
        net.train(); rng.shuffle(tr)
        for i in range(0, len(tr), bs):
            lp, lr_, lv, _, _ = loss(batch(tr[i:i + bs]))
            opt.zero_grad(); (lp + 10 * lr_ + 0.5 * lv).backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        net.eval()
        with torch.no_grad():
            lp, lr_, lv, la, acc = loss(Vb)
        print(f"epoch {ep:3d}  val pick acc {float(acc):.1%} (ce {float(lp):.3f})  residual mse {float(lr_):.4f}  "
              f"action mse {float(la):.4f} (explained {1 - float(la) / yvar:.1%})  value mse {float(lv):.3f}", flush=True)
    torch.save({k: v.cpu() for k, v in net.state_dict().items()}, out)
    print(f"{len(tr)} train / {len(val)} held-out episodes -> {out}")


def _eval_one(args):
    net_path, seed, hour = args
    import torch
    warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL); torch.set_num_threads(1)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            ep = fly(seed, hour, net=None if net_path == "teacher" else load(net_path),  # (teacher: RL3 + GNC, the paired base)
                     res=net_path != "teacher" and os.environ.get("E2E_RES") == "1")  # (E2E_RES=1: a residual net, §262)
    except Exception as e:
        return dict(seed=seed, error=repr(e))
    return dict(seed=seed, hour=hour, pops=ep["pops"], steps=len(ep["X"]))


if __name__ == "__main__":
    import multiprocessing as mp
    warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
    mode = sys.argv[1]
    if mode == "bc":  # dev/e2e.py bc DEMO_DIR OUT.pt [epochs]
        bc(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 40)
    if mode == "evalnet":  # dev/e2e.py evalnet NET.pt SEED_LO SEED_HI  (mean actions; hours cycle through E2E_HOURS)
        from dev.seed_registry import record as reg
        lo, hi = int(sys.argv[3]), int(sys.argv[4]); seeds = list(range(lo, hi + 1))
        reg(seeds, "e2e_eval", sys.argv[2], 40)
        jobs = [(sys.argv[2], s, HOURS[i % len(HOURS)]) for i, s in enumerate(seeds)]
        with mp.get_context("spawn").Pool(int(os.environ.get("PROCS", 4))) as p:
            res = p.map(_eval_one, jobs)
        ok = [r for r in res if "pops" in r]
        print(json.dumps(res))  # (per seed: paired comparisons, dev/e2e.py pair)
        print(f"{sys.argv[2]}: mean pops {np.mean([r['pops'] for r in ok]):.3f} over {len(ok)} seeds ({len(res) - len(ok)} failed)")
    if mode == "record":
        out, seeds = Path(sys.argv[2]), [int(s) for s in sys.argv[3:]]
        out.mkdir(parents=True, exist_ok=True)
        from dev.seed_registry import record as reg
        rng = np.random.default_rng(seeds[0])
        net_path = os.environ.get("E2E_NET", "")
        jobs = [(str(out), s, int(rng.choice(HOURS)), net_path) for s in seeds]
        reg(seeds, "e2e_record", f"imitation data {os.environ.get('E2E_KW', '')} net {net_path or '-'} -> {out}", 40)
        with mp.get_context("spawn").Pool(int(os.environ.get("PROCS", 1)), maxtasksperchild=1) as p:
            for line in p.imap_unordered(_record_one, jobs):
                print(line, flush=True)
    elif mode == "fly":
        import torch
        from dev.seed_registry import record as reg
        net = make_net() if sys.argv[2] == "random" else load(sys.argv[2])
        seed = int(sys.argv[3]); reg([seed], "e2e_fly", sys.argv[2], 40)
        with contextlib.redirect_stdout(io.StringIO()):
            ep = fly(seed, int(os.environ.get("E2E_HOUR", 21)), net=net)
        print(json.dumps(dict(seed=seed, pops=ep["pops"], steps=len(ep["X"]), launch=round(ep["launch"], 2),
                              u_mean=np.round(np.abs(ep["U"]).mean(0), 3).tolist() if len(ep["U"]) else None)))

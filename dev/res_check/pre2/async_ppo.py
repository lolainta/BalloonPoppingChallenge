"""Asynchronous PPO for RL2/RL3 guidance: every env runs on its own, nobody waits for the slowest env.

Workers (one process per env, any node) act with the numpy policy (agents/rl2_agent.py / rl3_agent.py),
send 64-step segments (obs, action, behaviour log-prob, value, reward, done, bootstrap value) to the learner
and reload weights between segments. The learner (GPU) keeps a receiver thread, runs a PPO update whenever
a batch is full, publishes the new weights to <run>/live_policy.npz, and drops segments that lag more than
MAX_LAG versions behind (the PPO ratio + clip covers the small lag that remains).

Learner (spawns LOCAL workers itself):
  .venv-gpu/bin/python dev/async_ppo.py learn --run runs/ppo8 --launch --init runs/ppo7/last.zip --local 94
Extra node:
  .venv/bin/python dev/async_ppo.py worker --run runs/ppo8 --n 30 --launch
Outputs match train_ppo2.py: progress.csv, <name>_<steps>.npz checkpoints (+ rl2_sl_<steps>.npz), best_episodes.csv.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse, csv, json, queue, shutil, socket, subprocess, sys, threading, time
from multiprocessing import Process
from multiprocessing.connection import Client, Listener, wait
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dev.remote_envs import authkey  # noqa: E402

SEG, MAX_LAG = 64, 2
GAMMA, LAM, CLIP, ENT, VF, LR, EPOCHS, MAXGRAD = 0.99, 0.95, 0.1, 0.003, 0.5, 1e-4, 4, 0.5
GAMMA = float(os.environ.get("RL3_GAMMA", GAMMA))  # per-decision discount; used only by the learner (GAE, SIL returns)
# self-imitation (Oh et al. 2018): workers send whole episodes with >= SIL_MIN pops; the learner adds
# -log pi(a|s) * max(R - V(s), 0) on them (R = discounted return-to-go). Off unless SIL_MIN is set.
SIL_MIN = float(os.environ.get("SIL_MIN", "inf"))
SIL_W, SIL_VF, SIL_BUF, SIL_BATCH = float(os.environ.get("SIL_W", 1.0)), 0.01, 300, 1024
# RL3: no discount on pre-launch (wait / launch) decisions. With 0.99 per decision, waiting 34 s on the pad (68
# decisions) halves the value of every later pop, so PPO prefers launching at once even when waiting pops more.
WAIT_NODISCOUNT = bool(os.environ.get("RL3_WAIT_NODISCOUNT"))
LAUNCHED_COL = 62  # rl2_agent.G_DIM: RL3's "launched" flag (0 on the pad)
# exploration where the policy picks a NEW target (post-launch, no token flagged as the current target): a learned
# SAC-style temperature alpha keeps the target head's entropy on those rows near RL3_NEWENT nats. It had collapsed to
# ~0.02-0.1 nats (top-1 0.96), so PPO never tried the better choices a simulator search finds (DECISION_LOG §139, §141).
NEWENT = float(os.environ.get("RL3_NEWENT", 0))
# the same for the speed-level (stretch) head on every post-launch decision (2026-09-29: its top-1 was 0.978, §143)
STRENT = float(os.environ.get("RL3_STRENT", 0))
LOG_ALPHA_MIN, LOG_ALPHA_MAX = float(np.log(1e-4)), float(np.log(0.1))  # the 0.35 run let alpha decay to ~0 after an
# entropy overshoot (1.3 nats), so it could not come back in time (§143): keep the temperatures in [1e-4, 0.1]
TOK0, TOK_F, TOK_CUR = LAUNCHED_COL + 2, 15, 8  # rl3_agent.G_DIM (first token column), rl2_agent.F, "is current target"


def gammas(obs):
    """Per-step discount: 1 for decisions taken on the pad (RL3_WAIT_NODISCOUNT), else GAMMA."""
    g = np.full(len(obs), GAMMA, np.float32)
    if WAIT_NODISCOUNT and len(obs) and np.asarray(obs).shape[1] > LAUNCHED_COL:
        g[np.asarray(obs)[:, LAUNCHED_COL] < 0.5] = 1.0
    return g


def dims(launch):
    from BalloonPoppingGymEnv.agents.rl3_agent import ATT
    return ([100, 4, 2] + ([len(ATT)] if os.environ.get("RL3_ATT") else [])) if launch else [100, 4]


ATTC = bool(os.environ.get("RL3_ATTC"))  # continuous launch attitude: 2 Gaussian action dims after the categorical ones
assert not (ATTC and os.environ.get("RL3_ATT")), "RL3_ATT (discrete) and RL3_ATTC (continuous) are exclusive"
# launch learned separately (user 2026-09-26): pad decisions (wait / launch / attitude) get the episode's pops minus their own
# baseline V_pad as advantage (a contextual bandit) instead of the flight GAE; the flight PPO loss skips pad rows.
# RL3_ATT_RAND = share of episodes whose launch attitude is drawn at random (the flight policy learns any launch;
# those attitudes are not the policy's, so they do not train the attitude head).
LAUNCH_SEP = bool(os.environ.get("RL3_LAUNCH_SEP"))
ATT_RAND = float(os.environ.get("RL3_ATT_RAND", 0.0))
LAUNCH_LAMBDA = float(os.environ.get("RL3_LAUNCH_LAMBDA", LAM))  # GAE lambda along the pad decisions (launch_loss)
LOG2PI = float(np.log(2 * np.pi))
# learned guidance residual (RL3_RES=<bound>, 2026-10-01): a Gaussian head on u (3,) after launch, the GNC adds
# bound * a_thrust * tanh(u) to its guidance force until the next decision (rl3_agent.set_res); u = the last 3 action columns
RES = bool(os.environ.get("RL3_RES"))


def one_width(rows):
    """All action rows equally wide (a reload that adds / drops a head between segments changes the width)."""
    return len({len(r) for r in rows}) <= 1


def res_gauss(net, mean, u, o):
    """Log-prob / entropy of the guidance residual u (RL3_RES), counted after launch only (pad rows: 0). The raw log std
    (a clamp here would zero its gradient at the bounds); the learner clamps the parameter after each step."""
    ls = net.res_logstd
    post = (o[:, LAUNCHED_COL] >= 0.5).float()
    lp = (-0.5 * ((u - mean) / ls.exp()) ** 2 - ls - 0.5 * LOG2PI).sum(-1)
    return lp * post, (ls + 0.5 * (1.0 + LOG2PI)).sum() * post


# ---------------------------------------------------------------------------------------------- worker
def _log_softmax(z):
    z = z - z.max()
    return z - np.log(np.exp(z).sum())


def act(agent, x, launch, rng):
    a, lp, v = _act(agent, x, launch, rng)
    if launch and "res.w" in agent.net:  # residual u ~ N(mean, std) after launch; on the pad 0 with log-prob 0
        from BalloonPoppingGymEnv.agents import rl2_agent as v2
        from BalloonPoppingGymEnv.agents.rl3_agent import RES_LS
        u = np.zeros(3, np.float32)
        if x[v2.G_DIM] >= 0.5:
            m, ls = agent.res_u, np.clip(agent.net["res.logstd"], *RES_LS)
            u = (m + np.exp(ls) * rng.standard_normal(3)).astype(np.float32)  # float32: as the learner sees it
            lp += float((-0.5 * ((u - m) / np.exp(ls)) ** 2 - ls - 0.5 * LOG2PI).sum())
        a = np.concatenate([a.astype(float), u])
    return a, lp, v


def _act(agent, x, launch, rng):
    """Sample an action with the numpy policy; phase masking matches dev/rl3_net.py (fixed parts: log-prob 0)."""
    if launch:
        from BalloonPoppingGymEnv.agents import rl2_agent as v2
        tl, sl, ll, v, al = agent.forward(x)
        att = "att.w" in agent.net
        if x[v2.G_DIM] < 0.5:  # pre-launch: only the launch decision (and attitude) is free
            lp = _log_softmax(ll); j = rng.choice(2, p=np.exp(lp))
            if "attc.w" in agent.net:  # Gaussian on u (mean al, learned std)
                ls = agent.net["attc.logstd"]; u = al + np.exp(ls) * rng.standard_normal(2)
                glp = float((-0.5 * ((u - al) / np.exp(ls)) ** 2 - ls - 0.5 * LOG2PI).sum())
                agent.lp_parts = (float(lp[j]), glp)
                return np.array([0, 3, j, u[0], u[1]]), float(lp[j]) + glp, float(v)
            if not att:
                agent.lp_parts = (float(lp[j]), 0.0)
                return np.array([0, 3, j]), float(lp[j]), float(v)
            la = _log_softmax(al); k = rng.choice(len(al), p=np.exp(la))
            agent.lp_parts = (float(lp[j]), float(la[k]))
            return np.array([0, 3, j, k]), float(lp[j] + la[k]), float(v)
        parts = [tl, sl]
    else:
        tl, sl, v = agent.forward(x)
        parts = [tl, sl]
    a, logp = [], 0.0
    for z in parts:
        lp = _log_softmax(z); j = rng.choice(len(z), p=np.exp(lp))
        a.append(j); logp += float(lp[j])
    if launch:
        a += [0, 0.0, 0.0] if "attc.w" in agent.net else ([0, 0] if att else [0])
    return np.array(a, float if "attc.w" in agent.net else int), logp, float(v)


def worker_loop(run, launch, rank):
    import warnings, logging
    warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
    if launch:
        from dev.rl3_env import GuidanceEnv3 as E
        from BalloonPoppingGymEnv.agents.rl3_agent import RL3Agent as A
    else:
        from dev.rl2_env import GuidanceEnv as E
        from BalloonPoppingGymEnv.agents.rl2_agent import RL2Agent as A
    run = Path(run)

    def connect():  # (re)connect to whatever learner is published; survives learner restarts
        while True:
            try:
                host, port = (run / "learner.addr").read_text().split()
                return Client((host, int(port)), authkey=authkey())
            except (OSError, ValueError, EOFError):
                time.sleep(5)

    conn = connect()

    def send(msg):
        nonlocal conn
        while True:
            try:
                conn.send(msg); return
            except (OSError, EOFError):
                conn = connect()
    rng = np.random.default_rng((os.getpid() * 7919 + rank) % 2**32)
    env = E(seed_lo=100_000, seed_hi=10_000_000)
    live = run / "live_policy.npz"
    agent, mtime, version = A.__new__(A), None, -1

    def reload():
        nonlocal mtime, version
        m = live.stat().st_mtime
        if m != mtime:
            d = dict(np.load(live))
            version, agent.net, mtime = int(d.pop("version")), d, m
            if launch:  # fly with the learner's GNC tunables (gnc.* in its weights; none -> defaults), not env vars
                from BalloonPoppingGymEnv.agents.rl2_agent import GNC_TUNABLES
                for k in GNC_TUNABLES:
                    env.agent_kw.pop(k, None)
                env.agent_kw.update({k[4:]: float(v) for k, v in d.items() if k.startswith("gnc.")})
                env.agent_kw.pop("res_bound", None)
                if "res.bound" in d:
                    env.agent_kw["res_bound"] = float(d["res.bound"])

    while not live.exists():
        time.sleep(2)
    reload()
    x, info = env.reset(seed=int(rng.integers(2**31)))
    ep_r, ep = 0.0, {"obs": [], "act": [], "rew": []}
    PAD_KEYS = ("obs", "act", "lpl", "lpa", "attv")
    pad, rand_att = {k: [] for k in PAD_KEYS + ("went",)}, rng.random() < ATT_RAND
    # LAUNCHDATA=dir: log each episode's launch row (obs, launch action, pops, random-attitude flag) for an
    # independent launch-value model (user 2026-09-26); 5 episodes per file, ~6 KB per episode
    ldir, lrow, lbuf_w = os.environ.get("LAUNCHDATA"), None, []
    while True:
        try:
            reload()
        except Exception:
            pass  # file being replaced: keep the current weights for one more segment
        seg = {k: [] for k in ("obs", "act", "logp", "val", "rew", "done")}
        v_seg = version
        for _ in range(SEG):
            a, lp, v = act(agent, x, launch, rng)
            on_pad = launch and LAUNCH_SEP and x[LAUNCHED_COL] < 0.5
            if on_pad:  # a pad decision: kept for the launch update
                attv = 1.0
                if rand_att and len(a) > 3 + 3 * RES:  # random-attitude episode: every pad row carries it (also a forced launch at T_MAX)
                    if ATTC:
                        e = rng.uniform(56.0, 89.5); off = rng.uniform(-170.0, 170.0)
                        a = a.astype(float); a[3] = np.log((e - 55.0) / (90.0 - e)); a[4] = np.arctanh(off / 180.0)
                    else:
                        from BalloonPoppingGymEnv.agents.rl3_agent import ATT
                        a = a.copy(); a[3] = rng.integers(len(ATT))
                    attv = 0.0
                for k, val in zip(PAD_KEYS, (x, a, *agent.lp_parts, attv)):
                    pad[k].append(val)
            x2, r, done, _, inf = env.step(a)
            if on_pad:
                pad["went"].append(float(env.ag.launched()))  # this row launched (chosen, or forced at T_MAX)
            if ldir and launch and x[LAUNCHED_COL] < 0.5 and env.ag.launched():
                lrow = (np.asarray(x, np.float32), np.asarray(a, np.float32), float(rand_att))
            for k, val in zip(seg, (x, a, lp, v, r, done)):
                seg[k].append(val)
            ep_r += r
            if SIL_MIN < np.inf:
                ep["obs"].append(x); ep["act"].append(a); ep["rew"].append(r)
            if done:
                send(("episode", {"r": ep_r, "seed": inf.get("seed"), "last_pop": inf.get("last_pop"), "version": v_seg}))
                if LAUNCH_SEP and pad["obs"] and one_width(pad["act"]):
                    send(("launch", {**{k: np.array(v, np.float32) for k, v in pad.items()}, "R": float(ep_r), "version": v_seg}))
                if ldir and lrow is not None:
                    lbuf_w.append((*lrow, float(ep_r))); lrow = None
                    if len(lbuf_w) >= 5:
                        os.makedirs(ldir, exist_ok=True)
                        np.savez(os.path.join(ldir, f"{os.getpid()}_{time.time_ns()}.npz"), obs=np.stack([b[0] for b in lbuf_w]),
                                 act=np.stack([np.resize(b[1], 5) for b in lbuf_w]), rand_att=np.array([b[2] for b in lbuf_w], np.float32),
                                 R=np.array([b[3] for b in lbuf_w], np.float32), version=np.int32(v_seg))
                        lbuf_w = []
                pad, rand_att = {k: [] for k in PAD_KEYS + ("went",)}, rng.random() < ATT_RAND
                if ep_r >= SIL_MIN and one_width(ep["act"]):  # (not an episode across a width-changing reload)
                    send(("demo", {"obs": np.array(ep["obs"], np.float32), "act": np.array(ep["act"]), "rew": np.array(ep["rew"], np.float32)}))
                ep_r, ep = 0.0, {"obs": [], "act": [], "rew": []}
                x2, _ = env.reset()
            x = x2
        boot = 0.0 if seg["done"][-1] else act(agent, x, launch, rng)[2]
        if one_width(seg["act"]):  # else dropped: it straddles a width-changing reload
            send(("segment", v_seg, {k: np.array(v) for k, v in seg.items()}, boot))


def worker_main(run, n, launch, rank0=0):
    from dev.rl2_env import clean_stale_tmp
    clean_stale_tmp()
    ps = [Process(target=worker_loop, args=(run, launch, rank0 + i), daemon=True) for i in range(n)]
    for p in ps:
        p.start()
    for p in ps:
        p.join()


# ---------------------------------------------------------------------------------------------- learner
def learn(a):
    import torch
    from stable_baselines3 import PPO
    if a.launch:
        from dev.rl3_net import Net3 as Net, export
    else:
        from dev.rl2_net import Net, export
    from dev.train_ppo2 import BestEpisodes

    run = Path(a.run); run.mkdir(parents=True, exist_ok=True)
    name = "rl3_policy" if a.launch else "rl2_policy"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu":
        torch.set_num_threads(4)  # OMP_NUM_THREADS=1 is for the env workers; the learner needs a few threads
    assert a.launch or not RES, "RL3_RES needs --launch"
    from BalloonPoppingGymEnv.agents.rl3_agent import RES_LS
    net = Net().to(dev)
    newp = [p for n, p in net.named_parameters() if n.startswith("res")]  # RL3_RES head: its own param group (last)
    opt = torch.optim.Adam([p for n, p in net.named_parameters() if not n.startswith("res")], lr=a.lr)
    log_alpha = torch.tensor(float(np.log(0.02)), device=dev, requires_grad=True)  # RL3_NEWENT temperature
    log_alpha_s = torch.tensor(float(np.log(0.02)), device=dev, requires_grad=True)  # RL3_STRENT temperature
    aopt = torch.optim.Adam([log_alpha, log_alpha_s], lr=3e-3)

    def gauss(mean, u, o):
        """Log-prob / entropy of the Gaussian launch attitude u (RL3_ATTC), counted on the pad only (launched flag 0)."""
        ls = net.attc_logstd
        pre = (o[:, LAUNCHED_COL] < 0.5).float()
        lp = (-0.5 * ((u - mean) / ls.exp()) ** 2 - ls - 0.5 * LOG2PI).sum(-1)
        return lp * pre, (ls + 0.5 * (1.0 + LOG2PI)).sum() * pre
    rgauss = lambda mean, u, o: res_gauss(net, mean, u, o)  # noqa: E731
    steps, version = 0, 0
    ck = run / "async_last.pt"
    if newp and not ck.exists():
        opt.add_param_group({"params": newp})
    if ck.exists():
        s = torch.load(ck, map_location=dev)
        add_res = newp and not any(k.startswith("res") for k in s["net"])  # saved without the RL3_RES head
        r = net.load_state_dict(s["net"], strict=not add_res)
        assert not add_res or (not r.unexpected_keys and all(k.startswith("res") for k in r.missing_keys)), r
        if newp and not add_res:
            opt.add_param_group({"params": newp})
        opt.load_state_dict(s["opt"]); steps, version = s["steps"], s["version"]
        if add_res:  # keep the optimizer state of the old parameters; the new head starts at zero with a fresh one
            opt.add_param_group({"params": newp}); print("res head added (zero init)", flush=True)
        if s.get("log_alpha") is not None:
            log_alpha.data.fill_(float(s["log_alpha"]))
        if s.get("log_alpha_s") is not None:
            log_alpha_s.data.fill_(float(s["log_alpha_s"]))
        for g in opt.param_groups:  # --lr wins over the learning rate stored in the checkpoint
            g["lr"] = a.lr
        print(f"resumed {ck} at {steps} steps", flush=True)
    elif a.init_npz:  # exported numpy weights (fresh optimizer)
        from dev.rl3_net import load_npz
        load_npz(net, a.init_npz); net.to(dev)
        print(f"initialised from {a.init_npz}", flush=True)
    elif a.init:
        src = PPO.load(a.init, device="cpu")
        net.load_state_dict(src.policy.net.state_dict()); steps = int(src.num_timesteps)
        print(f"initialised from {a.init} ({steps} steps)", flush=True)
    if a.launch and os.environ.get("RL3_FREE_LAUNCH"):  # launch time learned freely: no time prior, fixed at 0
        net.lprior.requires_grad_(False)
        if a.init_npz and not ck.exists() and not os.environ.get("RL3_KEEP_LAUNCH"):  # fresh branch: forget
            # "launch at the first chance" (~18% per decision); RL3_KEEP_LAUNCH=1 keeps the learned launch head (tuning)
            with torch.no_grad():
                net.lprior.zero_(); net.lau.weight.zero_(); net.lau.bias.copy_(torch.tensor([0.0, -1.5], device=dev))
            print("free launch: lprior = 0 (frozen), launch head reset", flush=True)

    # a fresh branch starts V_pad (not exported) from scratch: fit it for RL3_LAUNCH_WARM updates before its
    # advantages may move the launch policy
    lwarm = version + int(os.environ.get("RL3_LAUNCH_WARM", 5)) if LAUNCH_SEP and a.init_npz and not ck.exists() else 0

    def publish():
        tmp = run / "live_policy.tmp.npz"
        export(net, tmp)
        d = dict(np.load(tmp)); d["version"] = version
        np.savez(tmp, **d); os.replace(tmp, run / "live_policy.npz")

    publish()
    lis = Listener(("0.0.0.0", 0), backlog=512, authkey=authkey())
    (run / "learner.addr").write_text(f"{socket.gethostname().split('.')[0]} {lis.address[1]}\n")
    inbox, conns = queue.Queue(), []

    def accept():
        while True:
            try:
                c = lis.accept()
                # a worker whose node dies mid-message (e.g. a preempted devq_low job) left a half-open TCP connection,
                # and receive() blocked in c.recv() forever: no segment reached the learner for 1.5 h (2026-09-29).
                # Keepalive makes the kernel drop a dead peer after ~60 s, so recv raises and the connection is removed.
                try:
                    k = socket.fromfd(c.fileno(), socket.AF_INET, socket.SOCK_STREAM)  # a dup of the same socket
                    k.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                    k.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 30)
                    k.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 10)
                    k.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 3)
                    k.setsockopt(socket.IPPROTO_TCP, socket.TCP_USER_TIMEOUT, 60000)
                    k.close()
                except OSError as e:
                    print(f"keepalive setup failed: {e!r}", flush=True)
                conns.append(c)
            except Exception as e:  # one reset/failed handshake must not stop accepting the rest
                print(f"accept failed: {e!r}", flush=True)

    def receive():
        while True:
            if not conns:
                time.sleep(0.5)
                continue
            for c in wait(list(conns), timeout=1.0):
                try:
                    inbox.put(c.recv())
                except Exception as e:  # EOF / reset / timeout / a corrupt message: drop that worker, keep receiving
                    if not isinstance(e, (EOFError, OSError)):
                        print(f"receive failed: {e!r}", flush=True)
                    if c in conns:
                        conns.remove(c)

    threading.Thread(target=accept, daemon=True).start()
    threading.Thread(target=receive, daemon=True).start()
    local = subprocess.Popen([sys.executable, __file__, "worker", "--run", str(run), "--n", str(a.local)] + (["--launch"] if a.launch else []))

    best = BestEpisodes(run, lambda n, p: export(n, p), name)
    prog = csv.writer(f := open(run / "progress.csv", "a", newline="")); t0 = t_ck = time.time()
    eps, sizes, steps0 = [], dims(a.launch), steps
    aw = len(sizes) + ((2 if ATTC else 0) + (3 if RES else 0) if a.launch else 0)  # action width of this run's policy
    wide = lambda m: np.shape(m["act"])[1:] == (aw,)  # noqa: E731  (stale workers of a run whose heads changed: other widths)
    # every finished training episode's env seed (organisers' seed record, 2026-09-29); written once per update
    seedw = csv.writer(sf := open(run / "episode_seeds.csv", "a", newline=""))
    if sf.tell() == 0:
        seedw.writerow(["seed", "version", "pops", "last_pop", "time"])
    seed_rows = []
    hist = []  # (time, steps, episodes so far) for recent rates
    n_eps_total = 0
    demos, n_demo = [], 0  # own good episodes: [(obs, act, return-to-go)], newest last
    tdemos, tseen = [], set()  # teacher episodes dropped into <run>/demos_in/*.npz (dev/search_teacher.py)

    def to_demo(d):
        R = np.zeros(len(d["rew"]), np.float32); g = 0.0; gam = gammas(d["obs"])
        for t in reversed(range(len(R))):
            g = float(d["rew"][t]) + gam[t] * g; R[t] = g
        return (np.asarray(d["obs"], np.float32), np.asarray(d["act"]), R)

    def pick(buf, n):
        lens = np.array([len(d[0]) for d in buf]); k = np.random.randint(0, lens.sum(), n)
        cum = np.cumsum(lens); e = np.searchsorted(cum, k, side="right"); i = k - (cum[e] - lens[e])
        return [buf[a][0][b] for a, b in zip(e, i)], [buf[a][1][b] for a, b in zip(e, i)], [buf[a][2][b] for a, b in zip(e, i)]

    def sil_batch():  # half own episodes, half teacher episodes (when both exist)
        parts = [pick(b, SIL_BATCH // 2 if demos and tdemos else SIL_BATCH) for b in (demos, tdemos) if b]
        T = lambda z, dt=torch.float32: torch.tensor(np.stack(z), dtype=dt, device=dev)
        return tuple(T(sum((q[j] for q in parts), []), dt) for j, dt in ((0, torch.float32), (1, torch.float32), (2, torch.float32)))  # actions: float (continuous dims); categorical ones are cast in the loss
    lbuf = []  # launch samples (RL3_LAUNCH_SEP): one per finished episode

    def launch_loss(k0):
        """PPO-clip on the pad decisions + V_pad regression; k0 = first column after the target / stretch logits
        (launch logits, then the attitude part). Advantages: GAE(lambda) along each episode's pad decisions, then the
        usual mean / std normalisation (= the flight PPO's treatment, with its own critic V_pad):
        delta_j = (pops if the rocket left the pad after row j else V_pad(s_j+1)) - V_pad(s_j), no discount.
        (2026-09-26: the old per-row "pops - V_pad" drove the launch time to an extreme, 1 s in scenario 4 and 45 s in
        scenario 1; a per-episode V_pad error hits all the wait rows of that episode at once.)
        Random-attitude episodes (attv 0) are left out: their pops say nothing about the timing."""
        L = [d for d in lbuf if d["version"] >= version - 2 * MAX_LAG and len(d["obs"]) and d["attv"].min() > 0]
        if sum(len(d["obs"]) for d in L) < 2:
            return 0.0
        pick, n = [], 0
        for i in np.random.permutation(len(L)):  # whole episodes (TD needs each episode's rows in order)
            pick.append(L[i]); n += len(L[i]["obs"])
            if n >= 1024:
                break
        cat = lambda k: np.concatenate([d[k] for d in pick])
        Ol = torch.tensor(cat("obs"), device=dev); Al = torch.tensor(cat("act"), device=dev)
        R = torch.tensor(np.concatenate([np.full(len(d["obs"]), d["R"], np.float32) for d in pick]), device=dev)
        lpl0, lpa0 = torch.tensor(cat("lpl"), device=dev), torch.tensor(cat("lpa"), device=dev)
        went = torch.tensor(cat("went"), device=dev)
        # V_pad reads the shared features but does not train them: a fresh V_pad (output ~0 vs ~8 pops) otherwise
        # sends huge regression gradients into the trunk and wrecks the policy in the first updates (all launch-sep
        # branches dropped right away, s4sepgae even during the V_pad warm-up, 2026-09-26)
        out, _, pgf = net(Ol, need_pg=True); vp = net.vl(pgf.detach())[:, 0]
        out = out[:, :-3] if RES else out  # the residual is not a pad decision
        v = vp.detach()
        nxt = torch.cat([v[1:], v[-1:]])  # V_pad of the next pad decision
        last = torch.zeros_like(v); last[np.cumsum([len(d["obs"]) for d in pick]) - 1] = 1.0  # never bootstrap across episodes
        delta = (torch.where((went > 0.5) | (last > 0.5), R, nxt) - v).cpu().numpy()
        adv, j = np.zeros(len(delta), np.float32), 0
        for d in pick:  # lambda-return advantages per episode, backwards
            k = len(d["obs"]); acc = 0.0
            for i in range(j + k - 1, j - 1, -1):
                acc = delta[i] + LAUNCH_LAMBDA * acc; adv[i] = acc
            j += k
        adv = torch.tensor((adv - adv.mean()) / (adv.std() + 1e-8), device=dev)  # the mean shift cancels a constant V_pad bias
        zl = torch.log_softmax(out[:, k0: k0 + 2], -1)
        lpl = zl.gather(1, Al[:, 2:3].long())[:, 0]
        go = went  # the attitude only matters on the row that launched (random-attitude episodes are out)
        if ATTC:
            lpa = gauss(out[:, k0 + 2: k0 + 4], Al[:, 3:5], Ol)[0]
            enta = net.attc_logstd.sum()
        elif out.shape[1] > k0 + 2:
            za = torch.log_softmax(out[:, k0 + 2:], -1); lpa = za.gather(1, Al[:, 3:4].long())[:, 0]
            enta = -(za.exp() * za).sum(-1).mean()
        else:
            lpa, enta = torch.zeros_like(lpl), torch.tensor(0.0, device=dev)
        clipped = lambda ratio: -torch.min(ratio * adv, ratio.clamp(1 - CLIP, 1 + CLIP) * adv)
        pg_l = clipped(torch.exp(lpl - lpl0)).mean()
        pg_a = (clipped(torch.exp(lpa - lpa0)) * go).sum() / go.sum().clamp(min=1)
        entl = -(zl.exp() * zl).sum(-1).mean()
        pol = float(version >= lwarm)  # V_pad warm-up on a fresh branch
        return pol * (pg_l + pg_a - ENT * (entl + enta)) + VF * (vp - R).pow(2).mean()

    try:
        while True:
            segs, n, dropped, lags = [], 0, 0, []
            while n < a.batch:
                msg = inbox.get()
                if msg[0] == "demo":
                    if not wide(msg[1]):
                        continue
                    demos.append(to_demo(msg[1])); demos = demos[-SIL_BUF:]; n_demo += 1
                    continue
                if msg[0] == "launch":
                    if wide(msg[1]):
                        lbuf.append(msg[1])
                    continue
                if msg[0] == "episode":
                    eps.append(msg[1]); eps = eps[-100:]; n_eps_total += 1
                    ep = msg[1]
                    seed_rows.append([ep["seed"], ep["version"], ep["r"], ep["last_pop"], int(time.time())])
                    best.check(steps, [{"episode": {"r": ep["r"]}, "seed": ep["seed"], "last_pop": ep["last_pop"]}], net)
                else:
                    lags.append(max(version - msg[1], 0))
                    if version - MAX_LAG <= msg[1] <= version and wide(msg[2]):  # > version: acted with a pre-restart policy newer than the resumed checkpoint
                        segs.append(msg); n += SEG
                    else:
                        dropped += 1
            for fn in sorted((run / "demos_in").glob("*.npz")) if (run / "demos_in").exists() else []:
                if fn.name not in tseen and not fn.name.startswith("."):
                    tseen.add(fn.name)
                    try:
                        d = dict(np.load(fn))
                        if np.asarray(d["act"]).shape[1] != 3 + (2 if ATTC else 1 if os.environ.get("RL3_ATT") else 0) + 3 * RES:
                            raise ValueError(f"action width {np.asarray(d['act']).shape[1]} does not match this run")
                        tdemos.append(to_demo(d)); tdemos = tdemos[-3000:]
                    except Exception as e:
                        print(f"bad teacher demo {fn.name}: {e!r}", flush=True)
            # GAE per segment on behaviour values
            obs, act_, oldlp, adv, ret = [], [], [], [], []
            for _, _, s, boot in segs:
                v = np.append(s["val"], boot); d = s["done"].astype(np.float32); r = s["rew"]
                A = np.zeros(SEG, np.float32); g = 0.0; gam = gammas(s["obs"]) if a.launch else np.full(SEG, GAMMA)
                for t in reversed(range(SEG)):
                    nonterm = 1.0 - d[t]
                    delta = r[t] + gam[t] * v[t + 1] * nonterm - v[t]
                    g = delta + gam[t] * LAM * nonterm * g
                    A[t] = g
                obs.append(s["obs"]); act_.append(s["act"]); oldlp.append(s["logp"]); adv.append(A); ret.append(A + s["val"])
            T = lambda z, dt=torch.float32: torch.tensor(np.concatenate(z), dtype=dt, device=dev)
            O, ACT, OLP, ADV, RET = T(obs), T(act_), T(oldlp), T(adv), T(ret)  # float: continuous dims ride along
            ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
            N = len(O)
            ent_log, sent_log = [], []
            for _ in range(EPOCHS):
                for j in torch.randperm(N, device=dev).split(N // 4):
                    out = net(O[j])
                    logits, val = (torch.cat(out[:2], -1), out[2]) if not a.launch else out
                    lp, ent, k = 0.0, 0.0, 0
                    for i, m in enumerate(sizes):
                        z = torch.log_softmax(logits[:, k: k + m], -1); k += m
                        lp = lp + z.gather(1, ACT[j, i: i + 1].long())[:, 0]
                        hi = -(z.exp() * z).sum(-1)
                        ent = ent + hi
                        if i == 0:
                            h_tgt = hi
                        elif i == 1:
                            h_str = hi
                    if ATTC:  # Gaussian launch attitude, only on the pad (it is ignored after launch)
                        glp, gent = gauss(logits[:, k: k + 2], ACT[j, len(sizes): len(sizes) + 2], O[j])
                        lp, ent = lp + glp, ent + gent
                    if RES:  # guidance residual: last 3 logits columns = its mean, last 3 action columns = u
                        rlp, rent = rgauss(logits[:, -3:], ACT[j, -3:], O[j])
                        lp, ent = lp + rlp, ent + rent
                    ratio = torch.exp(lp - OLP[j])
                    pgr = -torch.min(ratio * ADV[j], ratio.clamp(1 - CLIP, 1 + CLIP) * ADV[j])
                    if LAUNCH_SEP:  # pad rows are trained by launch_loss(); the flight loss uses post-launch rows only
                        fw = (O[j][:, LAUNCHED_COL] >= 0.5).float()
                        pg, ent = (pgr * fw).sum() / fw.sum().clamp(min=1), ent * fw
                    else:
                        pg = pgr.mean()
                    loss = pg + VF * (val - RET[j]).pow(2).mean() - ENT * ent.mean()
                    a_loss = None
                    if NEWENT:
                        Oj = O[j]
                        newm = ((Oj[:, LAUNCHED_COL] >= 0.5)
                                & (Oj[:, TOK0:].reshape(len(Oj), -1, TOK_F)[:, :, TOK_CUR].max(1).values < 0.5)).float()
                        if newm.sum() > 0:
                            h_new = (h_tgt * newm).sum() / newm.sum()
                            loss = loss - log_alpha.exp().detach() * h_new
                            a_loss = log_alpha * (h_new.detach() - NEWENT)  # alpha grows while the entropy is below target
                            ent_log.append((float(h_new.detach()), float(newm.sum())))
                    if STRENT:
                        postm = (O[j][:, LAUNCHED_COL] >= 0.5).float()
                        if postm.sum() > 0:
                            h_s = (h_str * postm).sum() / postm.sum()
                            loss = loss - log_alpha_s.exp().detach() * h_s
                            a_s = log_alpha_s * (h_s.detach() - STRENT)
                            a_loss = a_s if a_loss is None else a_loss + a_s
                            sent_log.append((float(h_s.detach()), float(postm.sum())))
                    if LAUNCH_SEP:
                        loss = loss + launch_loss(sum(sizes[:2]))
                    if demos or tdemos:
                        Od, Ad, Rd = sil_batch()
                        ld, vd = net(Od); lpd, k2 = 0.0, 0
                        for i, m in enumerate(sizes):
                            z = torch.log_softmax(ld[:, k2: k2 + m], -1); k2 += m
                            lpd = lpd + z.gather(1, Ad[:, i: i + 1].long())[:, 0]
                        if ATTC:
                            lpd = lpd + gauss(ld[:, k2: k2 + 2], Ad[:, len(sizes): len(sizes) + 2], Od)[0]
                        if RES:
                            lpd = lpd + rgauss(ld[:, -3:], Ad[:, -3:], Od)[0]
                        gap = (Rd - vd).clamp(min=0, max=5)
                        sw = (Od[:, LAUNCHED_COL] >= 0.5).float() if LAUNCH_SEP else torch.ones_like(gap)
                        sil = -(lpd * gap.detach() * sw).sum() / sw.sum().clamp(min=1) + SIL_VF * 0.5 * gap.pow(2).mean()
                        loss = loss + SIL_W * sil
                    opt.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), MAXGRAD); opt.step()
                    if RES:
                        net.res_logstd.data.clamp_(*RES_LS)
                    if a_loss is not None:
                        aopt.zero_grad(); a_loss.backward(); aopt.step()
                        with torch.no_grad():
                            log_alpha.clamp_(LOG_ALPHA_MIN, LOG_ALPHA_MAX); log_alpha_s.clamp_(LOG_ALPHA_MIN, LOG_ALPHA_MAX)
            steps += N; version += 1
            lbuf = [d for d in lbuf if d["version"] >= version - 2 * MAX_LAG]
            publish()
            lag = np.mean([version - 1 - s[1] for s in segs])
            prog.writerow([steps, len(eps), f"{np.mean([e['r'] for e in eps]) if eps else float('nan'):.3f}", f"{time.time() - t0:.0f}"]); f.flush()
            seedw.writerows(seed_rows); sf.flush(); seed_rows = []
            print(f"v{version} steps {steps} batch {N} mean_r {np.mean([e['r'] for e in eps]) if eps else 0:.2f} lag {lag:.2f} "
                  + (f"launch_eps {len(lbuf)} " if LAUNCH_SEP else "") +
                  f"workers {len(conns)} sps {(steps - steps0) / max(time.time() - t0, 1):.0f} "
                  + (f"newent {np.average([e[0] for e in ent_log], weights=[e[1] for e in ent_log]) if ent_log else float('nan'):.3f} alpha {float(log_alpha.exp()):.4f} " if NEWENT else "")
                  + (f"strent {np.average([e[0] for e in sent_log], weights=[e[1] for e in sent_log]) if sent_log else float('nan'):.3f} alpha_s {float(log_alpha_s.exp()):.4f} " if STRENT else "") +
                  f"demos {n_demo}/{len(demos)} teacher {len(tdemos)} dropped {dropped}/{len(lags)} ({100 * dropped / max(len(lags), 1):.1f}%) lag_hist {np.bincount(np.minimum(lags, 5), minlength=6).tolist()}", flush=True)
            now = time.time(); hist.append((now, steps, n_eps_total)); hist = [h for h in hist if now - h[0] <= 300]
            span = max(now - hist[0][0], 1e-6)
            status = {"version": version, "steps": steps, "workers": len(conns), "lag": round(float(lag), 2),
                      "dropped_pct": round(100 * dropped / max(len(lags), 1), 1), "lag_hist": np.bincount(np.minimum(lags, 5), minlength=6).tolist(),
                      "dec_per_s": round((steps - hist[0][1]) / span, 1), "eps_per_h": round(3600 * (n_eps_total - hist[0][2]) / span),
                      "updated": int(now)}
            if NEWENT and ent_log:
                status.update(newent=round(float(np.average([e[0] for e in ent_log], weights=[e[1] for e in ent_log])), 4),
                              alpha=round(float(log_alpha.exp()), 5))
            if STRENT and sent_log:
                status.update(strent=round(float(np.average([e[0] for e in sent_log], weights=[e[1] for e in sent_log])), 4),
                              alpha_s=round(float(log_alpha_s.exp()), 5))
            (run / "status.json").write_text(json.dumps(status))
            if time.time() - t_ck >= a.ckpt_every:
                t_ck = time.time()
                torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "steps": steps, "version": version,
                            "log_alpha": float(log_alpha.detach()) if NEWENT else None,
                            "log_alpha_s": float(log_alpha_s.detach()) if STRENT else None}, ck)
                export(net, run / f"{name}_{steps}.npz")
                sl = os.environ.get("SLMODEL")
                if sl and os.path.exists(sl):
                    shutil.copy2(sl, run / f"rl2_sl_{steps}.npz")
    finally:
        local.terminate()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["learn", "worker"])
    ap.add_argument("--run", required=True)
    ap.add_argument("--launch", action="store_true")
    ap.add_argument("--init", help="SB3 zip to start from (ignored if <run>/async_last.pt exists)")
    ap.add_argument("--local", type=int, default=8, help="learner: local worker processes")
    ap.add_argument("--batch", type=int, default=9600, help="transitions per PPO update")
    ap.add_argument("--n", type=int, default=8, help="worker: env processes on this node")
    ap.add_argument("--init_npz", help="exported numpy weights to start from (RL3)")
    ap.add_argument("--ckpt_every", type=int, default=600, help="seconds between checkpoints")
    ap.add_argument("--lr", type=float, default=LR)
    a = ap.parse_args()
    learn(a) if a.mode == "learn" else worker_main(a.run, a.n, a.launch)

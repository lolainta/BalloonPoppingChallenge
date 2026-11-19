"""B (from dev/tc2_check/check_c.py): a non-zero fast checkpoint (8-output res + tc + tc2 + fast, k from the npz) on one scenario-4
training world (dev/rl3_env.py + async_ppo.act, worker reload settings), flown 3 times:
 (ii) greedy sampling (argmax / zero noise): every post-launch row is FULL exactly when the evaluation agent decides there
      (needs_decision) and FAST exactly when its fast_tick fires; the target / stretch / residual / s_mul / n_mul / dv the training env
      applies (env.ag after step) equal what RL3Agent evaluation applies (FULL: choose + set_res; FAST: forward + set_res(u_full, fast_u));
 (iii) stochastic (the fast policy, then the same with the fast head zeroed): worker log-prob vs the learner's recomputation
      (async_ppo.learn's formulas: tgt = ACT[:, 0], gauss, res_gauss, fast_gauss) on pad / FULL / FAST rows, CPU float32 / float64
      (+ the worker's formulas in float64: the float32 floor);  (iv) a policy surrogate on those rows sends gradients into fast / fastc /
      fast_logstd (also from the zero head), none into them from FULL rows only, none into res / tc / tc2 from FAST rows only.
An n-output fast head (RL3_FAST_DIMS=n, RL3_FAST_STD=s as the npz was made): FAST rows' u columns n.. must be exactly 0, the log std
clamp's lower bound the npz's fast.lsmin (worker) = Net3.fast_lsmin (learner).
source dev/fast_check/env.sh; RL3_FAST=10 [RL3_FAST_DIMS=3 RL3_FAST_STD=0.03] PREFETCH=0 RL3_REUSE=10 $PY dev/fast_check/check_b.py NPZ [seed]"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import copy, sys, warnings
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dev import async_ppo as ap, rl2_env, rl3_env, rl3_net  # noqa: E402
from dev.rl3_env import GuidanceEnv3  # noqa: E402
from dev.rl3_net import Net3, load_npz  # noqa: E402
from BalloonPoppingGymEnv.agents import rl3_agent  # noqa: E402
from BalloonPoppingGymEnv.agents.rl2_agent import STRETCH  # noqa: E402
from BalloonPoppingGymEnv.agents.rl3_agent import RES_LS, RL3Agent  # noqa: E402

for m in (ap, rl2_env, rl3_env, rl3_net, rl3_agent):
    assert Path(m.__file__).resolve().is_relative_to(ROOT), m.__file__
warnings.filterwarnings("ignore")
torch.set_num_threads(2)
P = dict(np.load(sys.argv[1])); assert np.abs(P["fast.w"]).max() > 0 and len(P["res.b"]) == ap.NR == 8 and ap.FAST == int(P["fast.k"]) \
    and P["fast.w"].shape[1] == ap.NF
env = GuidanceEnv3(seed_lo=100_000, seed_hi=10_000_000)
env.agent_kw.update({k[4:]: float(v) for k, v in P.items() if k.startswith("gnc.")})  # (= worker_loop.reload)
env.agent_kw.update({f"res_{k}": float(P[f"res.{k}"]) for k in ("bound", "sbound", "nbound", "vbound") if f"res.{k}" in P})
env.agent_kw["fast_k"] = int(P["fast.k"])
agent = RL3Agent.__new__(RL3Agent); agent.net = P
NH = 5  # action columns before the flag: target, stretch, launch, attc (2)


class Greedy:  # the deterministic limit of the worker's sampling
    choice = staticmethod(lambda n, p: int(np.argmax(p)))
    standard_normal = staticmethod(lambda n: np.zeros(n))


def applied(g):
    return np.concatenate([g.res, [g.s_mul, g.n_mul], g.dv])


def episode(rng, seed=None, compare=False):
    x, _ = env.reset(seed=seed)
    X, A, LP, done, n = [], [], [], False, {"full": 0, "fast": 0}
    worst, w32, sched = {"full": 0.0, "fast": 0.0}, {"full": 0.0, "fast": 0.0}, 0
    while not done:
        a, lp, v = ap.act(agent, x, True, rng, env.fast)
        post = x[ap.LAUNCHED_COL] >= 0.5
        kind = "full" if a[NH] == 1 else "fast"
        if compare and post:
            ev = copy.deepcopy(env.ag); ev.net = P; ev.fast_k, ev.fast_off = int(P["fast.k"]), 0  # the evaluation agent in this state
            e_full = ev.needs_decision(env.obs); e_fast = not e_full and ev.fast_tick(env.obs)
            sched += (kind == "full") == e_full and (kind == "fast") == e_fast
            if kind == "full":
                i, s = ev.choose(env.obs); ev.set_res(ev.res_mean); e32 = copy.copy(ev); e32.set_res(np.float32(ev.res_mean))
            else:
                ev.forward(ev.features(env.obs)); e32 = copy.deepcopy(ev)
                ev.set_res(ev.u_full, ev.fast_u); e32.set_res(e32.u_full, np.float32(e32.fast_u))  # (a stored delta is float32)
        X.append(x); A.append(a); LP.append(lp)
        x, r, done, _, _ = env.step(a)
        if compare and post:
            g = env.ag
            if kind == "full":
                assert (int(a[0]), int(a[1])) == (i, s) and g.cmd[:2] == (i, STRETCH[s]), (a[:2], i, s, g.cmd)
            worst[kind] = max(worst[kind], np.abs(applied(g) - applied(ev)).max())
            w32[kind] = max(w32[kind], np.abs(applied(g) - applied(e32)).max())
            n[kind] += 1
    if compare:
        print(f"(ii) greedy episode: post-launch rows FULL {n['full']} / FAST {n['fast']}; row kind = the evaluation agent's schedule "
              f"(needs_decision / fast_tick) on {sched}/{n['full'] + n['fast']}; FULL target / stretch = evaluation's; applied res / s_mul "
              f"/ n_mul / dv max |train - eval|  FULL {worst['full']:.2e}  FAST {worst['fast']:.2e}  (eval's u / delta rounded to float32 "
              f"like a stored action: {w32['full']:.1e} / {w32['fast']:.1e}); pops {env.info.get('popped_count')}")
    return np.array(X, np.float32), np.array(A), np.array(LP)


def np_logp(Pn, X, A, padm):
    """Worker log-probs of the stored actions under weights Pn (= act()'s formulas)."""
    ag = RL3Agent.__new__(RL3Agent); ag.net = Pn
    ls_a, ls_r = Pn["attc.logstd"], np.clip(Pn["res.logstd"], *RES_LS)
    ls_f = np.clip(Pn["fast.logstd"], Pn.get("fast.lsmin", RES_LS[0]), RES_LS[1])
    gl = lambda u, m, ls: float((-0.5 * ((u - m) / np.exp(ls)) ** 2 - ls - 0.5 * ap.LOG2PI).sum())  # noqa: E731
    tot = np.zeros(len(X))
    for r, (x, a) in enumerate(zip(X, A)):
        tl, sl, ll, _, al = ag.forward(x)
        if padm[r]:
            tot[r] = ap._log_softmax(ll)[int(a[2])] + gl(a[3:5], al, ls_a)
        elif a[NH] == 0:
            tot[r] = gl(a[-ap.NR:][:ap.NF], ag.fast_u, ls_f)
        else:
            sl = ag.cond(int(a[0]), sl, tl)
            tot[r] = ap._log_softmax(tl)[int(a[0])] + ap._log_softmax(sl)[int(a[1])] + gl(a[-ap.NR:], ag.res_u, ls_r)
    return tot


def torch_logp(Pn, X, A, dt):
    """The learner's recomputation (async_ppo.learn: tgt = ACT[:, 0], gauss / rgauss / fast_gauss) -> (log-prob, entropy, net)."""
    path = Path(os.environ.get("TMPDIR", "/tmp")) / f"fast_check_{os.getpid()}.npz"; np.savez(path, **Pn)
    net = Net3(); load_npz(net, path); net.to("cpu", dt); path.unlink()
    O, ACT = torch.tensor(X, dtype=dt), torch.tensor(A, dtype=dt)
    logits, _ = net(O, tgt=ACT[:, 0].long())
    lp, ent, k = 0.0, 0.0, 0
    for i, m in enumerate(ap.dims(True)):
        z = torch.log_softmax(logits[:, k: k + m], -1); k += m
        lp = lp + z.gather(1, ACT[:, i: i + 1].long())[:, 0]; ent = ent - (z.exp() * z).sum(-1)
    ls = net.attc_logstd; pre = (O[:, ap.LAUNCHED_COL] < 0.5).to(dt)  # = learn.gauss
    lp = lp + (-0.5 * ((ACT[:, 3:5] - logits[:, k: k + 2]) / ls.exp()) ** 2 - ls - 0.5 * ap.LOG2PI).sum(-1) * pre
    rlp, rent = ap.res_gauss(net, logits[:, -ap.NR:], ACT[:, -ap.NR:], O)  # = learn.rgauss
    flp, fent = ap.fast_gauss(net, logits[:, -ap.NR - ap.NF: -ap.NR], ACT); full = ap.full_mask(ACT)  # (the learner: FULL rows today's heads, FAST rows these)
    lp, ent = torch.where(full, lp + rlp, flp), torch.where(full, ent + rent, fent)
    return lp, ent, net


seed = int(sys.argv[2]) if len(sys.argv) > 2 else 12345
zero = len(sys.argv) > 3 and sys.argv[3] == "zero"  # only (iii)/(iv) with the zero head, on a fresh world: a world's /tmp files
if not zero:  # are swept 30 min after its reset (rl2_env.clean_stale_tmp), and on a loaded machine this script takes longer
    episode(Greedy(), seed, compare=True)
P0 = dict(P, **{k: np.zeros_like(P[k]) for k in ("fast.w", "fast.b", "fastc.w", "fastc.b")})
for tag, Pn in (("zero fast head", P0),) if zero else (("fast", P),):
    agent.net = Pn
    X, A, LP = episode(np.random.default_rng(1), seed if zero else None)  # (fast: the same world again, RL3_REUSE=10: deep copies)
    padm = X[:, ap.LAUNCHED_COL] < 0.5
    fastm = ~padm & (A[:, NH] == 0); fullm = ~padm & ~fastm
    LPn = np_logp(Pn, X, A, padm)
    print(f"(iii) {tag}: stochastic episode rows {len(X)} (pad {int(padm.sum())}, FULL {int(fullm.sum())}, FAST {int(fastm.sum())}), "
          f"action width {A.shape[1]}, pops {env.info.get('popped_count')}; worker logp recomputed from the stored actions = stored: "
          f"max diff {np.abs(LPn - LP).max():.1e}; FAST rows' u columns {ap.NF}..{ap.NR - 1} max |.| "
          f"{np.abs(A[fastm][:, A.shape[1] - ap.NR + ap.NF:]).max(initial=0):.1e}, deltas max |.| {np.abs(A[fastm][:, -ap.NR:]).max(initial=0):.2e}; "
          f"fast log std lower bound worker {float(Pn.get('fast.lsmin', RES_LS[0])):.6f} learner {Net3().fast_lsmin:.6f} "
          f"(log 0.03 {np.log(0.03):.6f}), npz fast.logstd {Pn['fast.logstd']}")
    for dt in (torch.float32, torch.float64):
        lpt, ent, net = torch_logp(Pn, X, A, dt)
        d = np.abs(lpt.detach().numpy() - LP)
        print(f"    cpu {dt}: max |worker logp - learner logp|  pad {d[padm].max(initial=0):.2e}  FULL {d[fullm].max(initial=0):.2e}  FAST {d[fastm].max(initial=0):.2e}")
        if dt == torch.float64:  # the worker's formulas in float64 (weights / observations cast; stored actions are float32 values)
            P64 = {k: (v.astype(np.float64) if v.dtype == np.float32 else v) for k, v in Pn.items()}
            LP64 = np_logp(P64, X.astype(np.float64), A, padm)
            e = np.abs(LP64 - lpt.detach().numpy())
            print(f"    numpy float64 worker formulas vs learner float64: max diff pad {e[padm].max(initial=0):.2e}  FULL {e[fullm].max(initial=0):.2e}  "
                  f"FAST {e[fastm].max(initial=0):.2e}")
        if dt == torch.float32:  # (iv) PPO-like surrogates: all rows / FULL rows only / FAST rows only
            w = torch.tensor(np.random.default_rng(2).standard_normal(len(X)), dtype=dt)
            for rows, msk in (("all rows", np.ones(len(X), bool)), ("FULL+pad rows only", ~fastm), ("FAST rows only", fastm)):
                lpt, ent, net = torch_logp(Pn, X, A, dt)
                net.zero_grad(); (-(w * lpt * torch.tensor(msk, dtype=dt)).mean() - 0.003 * (ent * torch.tensor(msk, dtype=dt)).mean()).backward()
                gn = {n_: float(p.grad.norm()) if p.grad is not None else 0.0 for n_, p in net.named_parameters()}
                print(f"(iv) {tag}, {rows}: grad norms " + ", ".join(f"{k} {gn[k]:.2e}" for k in (
                    "fast.weight", "fast.bias", "fastc.weight", "fastc.bias", "fast_logstd", "res.weight", "res_logstd", "tc.weight",
                    "tc2.weight", "str.weight")))

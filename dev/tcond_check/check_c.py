"""C: a non-zero tc checkpoint on one scenario-4 training world (dev/rl3_env.py + async_ppo.act), flown 3 times:
 (ii) greedy sampling (argmax / zero noise): at every post-launch decision, the stretch and residual the training env applies
      (env.ag.cmd / res / s_mul / n_mul after step) equal what RL3Agent evaluation (choose + set_res) applies;
 (iii) stochastic (the tc policy, then the same with tc zeroed as the reference): worker log-prob vs the learner's recomputation (formulas of async_ppo.learn, tgt = ACT[:, 0]) on pad and
      post-launch rows, CPU float32 / float64; numpy vs torch conditioned stretch logits / residual mean;
 (iv) the learner's policy loss on those rows sends gradients into tc.
source dev/tcond_check/env.sh; PREFETCH=0 RL3_REUSE=3 $PY dev/tcond_check/check_c.py TC.npz [seed]"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import copy, sys, warnings
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dev import async_ppo as ap, rl3_env, rl3_net  # noqa: E402
from dev.rl3_env import GuidanceEnv3  # noqa: E402
from dev.rl3_net import Net3, load_npz  # noqa: E402
from BalloonPoppingGymEnv.agents import rl3_agent  # noqa: E402
from BalloonPoppingGymEnv.agents.rl2_agent import STRETCH  # noqa: E402
from BalloonPoppingGymEnv.agents.rl3_agent import RES_LS, RL3Agent  # noqa: E402

for m in (ap, rl3_env, rl3_net, rl3_agent):
    assert Path(m.__file__).resolve().is_relative_to(ROOT), m.__file__
warnings.filterwarnings("ignore")
torch.set_num_threads(2)
P = dict(np.load(sys.argv[1])); assert "tc.w" in P and np.abs(P["tc.w"]).max() > 0
env = GuidanceEnv3(seed_lo=100_000, seed_hi=10_000_000)
env.agent_kw.update({k[4:]: float(v) for k, v in P.items() if k.startswith("gnc.")})
env.agent_kw.update({f"res_{k}": float(P[f"res.{k}"]) for k in ("bound", "sbound", "nbound") if f"res.{k}" in P})
agent = RL3Agent.__new__(RL3Agent); agent.net = P


class Greedy:  # the deterministic limit of the worker's sampling
    choice = staticmethod(lambda n, p: int(np.argmax(p)))
    standard_normal = staticmethod(lambda n: np.zeros(n))


def episode(rng, seed=None, compare=False):
    x, _ = env.reset(seed=seed)
    X, A, LP, done, n_cmp, n_diff_s, worst, w32 = [], [], [], False, 0, 0, 0.0, 0.0
    while not done:
        a, lp, v = ap.act(agent, x, True, rng)
        post = x[ap.LAUNCHED_COL] >= 0.5
        if compare and post:
            ev = copy.deepcopy(env.ag); ev.net = P  # the evaluation agent in the same state
            i, s = ev.choose(env.obs); ev.set_res(ev.res_mean)
            e32 = copy.copy(ev); e32.set_res(np.float32(ev.res_mean))  # (a training action stores u in float32)
            s0 = int(np.argmax(ev.forward(ev.features(env.obs))[1]))  # the unconditioned speed gear
            n_diff_s += s0 != s
        X.append(x); A.append(a); LP.append(lp)
        x, r, done, _, _ = env.step(a)
        if compare and post:
            g = env.ag
            assert (int(a[0]), int(a[1])) == (i, s) and g.cmd[:2] == (i, STRETCH[s]), (a[:2], i, s, g.cmd)
            worst = max(worst, np.abs(g.res - ev.res).max(), abs(g.s_mul - ev.s_mul), abs(g.n_mul - ev.n_mul))
            w32 = max(w32, np.abs(g.res - e32.res).max(), abs(g.s_mul - e32.s_mul), abs(g.n_mul - e32.n_mul))
            n_cmp += 1
    if compare:
        print(f"(ii) greedy episode: {n_cmp} post-launch decisions, target / stretch identical to evaluation, applied residual "
              f"/ s_mul / n_mul max |train - eval| {worst:.2e} (eval mean rounded to float32 like a stored action: {w32:.1e}); "
              f"conditioning changed the speed gear at {n_diff_s}/{n_cmp}; pops {env.info.get('popped_count')}")
    return np.array(X, np.float32), np.array(A), np.array(LP)


def np_logp(Pn, X, A, padm):
    """Worker log-probs of the stored actions under weights Pn (= act()'s formulas) -> (total, residual part, conditioned
    stretch logits, conditioned residual mean) (the last two on post-launch rows)."""
    ag = RL3Agent.__new__(RL3Agent); ag.net = Pn
    ls_a, ls_r = Pn["attc.logstd"], np.clip(Pn["res.logstd"], *RES_LS)
    gl = lambda u, m, ls: float((-0.5 * ((u - m) / np.exp(ls)) ** 2 - ls - 0.5 * ap.LOG2PI).sum())  # noqa: E731
    tot, rp, SL, RU = np.zeros(len(X)), np.zeros(len(X)), [], []
    for r, (x, a) in enumerate(zip(X, A)):
        tl, sl, ll, _, al = ag.forward(x)
        if padm[r]:
            tot[r] = ap._log_softmax(ll)[int(a[2])] + gl(a[3:5], al, ls_a)
        else:
            sl = ag.cond(int(a[0]), sl); SL.append(sl); RU.append(ag.res_u)
            rp[r] = gl(a[-ap.NR:].astype(np.float32), ag.res_u, ls_r)
            tot[r] = ap._log_softmax(tl)[int(a[0])] + ap._log_softmax(sl)[int(a[1])] + rp[r]
    return tot, rp, np.array(SL), np.array(RU)


def torch_logp(Pn, X, A, dt):
    """The learner's recomputation (async_ppo.learn: tgt = ACT[:, 0], gauss / rgauss) -> (total, residual part, logits, net)."""
    path = Path(os.environ.get("TMPDIR", "/tmp")) / f"tc_check_{os.getpid()}.npz"; np.savez(path, **Pn)
    net = Net3(); load_npz(net, path); net.to("cpu", dt); path.unlink()
    O, ACT = torch.tensor(X, dtype=dt), torch.tensor(A, dtype=dt)
    logits, _ = net(O, tgt=ACT[:, 0].long())
    lp, k = 0.0, 0
    for i, m in enumerate(ap.dims(True)):
        z = torch.log_softmax(logits[:, k: k + m], -1); k += m
        lp = lp + z.gather(1, ACT[:, i: i + 1].long())[:, 0]
    ls = net.attc_logstd; pre = (O[:, ap.LAUNCHED_COL] < 0.5).to(dt)  # = learn.gauss
    lp = lp + (-0.5 * ((ACT[:, 3:5] - logits[:, k: k + 2]) / ls.exp()) ** 2 - ls - 0.5 * ap.LOG2PI).sum(-1) * pre
    rlp = ap.res_gauss(net, logits[:, -ap.NR:], ACT[:, -ap.NR:], O)[0]  # = learn.rgauss
    return lp + rlp, rlp, logits, net


seed = int(sys.argv[2]) if len(sys.argv) > 2 else 12345
episode(Greedy(), seed, compare=True)
P0 = dict(P, **{"tc.w": np.zeros_like(P["tc.w"]), "tc.b": np.zeros_like(P["tc.b"])})
for tag, Pn in (("tc", P), ("zero tc (reference: the same world, its own sampled episode)", P0)):
    agent.net = Pn
    X, A, LP = episode(np.random.default_rng(1))  # the same world again (RL3_REUSE=3, PREFETCH=0)
    padm = X[:, ap.LAUNCHED_COL] < 0.5
    LPn, RPn, SLn, RUn = np_logp(Pn, X, A, padm)
    print(f"(iii) {tag}: stochastic episode rows {len(X)} (pad {int(padm.sum())}), action width {A.shape[1]}, pops "
          f"{env.info.get('popped_count')}; worker logp recomputed from the stored actions = stored: max diff {np.abs(LPn - LP).max():.1e}")
    for dt in (torch.float32, torch.float64):
        lpt, rlp, logits, net = torch_logp(Pn, X, A, dt)
        assert bool(net.tc.weight.abs().max() > 0) == (Pn is P)
        lp, rlp = lpt.detach().numpy(), rlp.detach().numpy()
        d, dr = np.abs(lp - LP), np.abs(rlp - RPn)
        sl_t, ru_t = logits[:, 100:104].detach().numpy()[~padm], logits[:, -ap.NR:].detach().numpy()[~padm]
        print(f"    cpu {dt}: max |worker logp - learner logp|  pad {d[padm].max():.2e}  post-launch {d[~padm].max():.2e} "
              f"(residual part {dr[~padm].max():.2e}, categorical part {np.abs(d - dr)[~padm].max():.2e});  numpy vs torch "
              f"conditioned stretch logits {np.abs(sl_t - SLn).max():.2e}, residual mean {np.abs(ru_t - RUn).max():.2e}")
        if dt == torch.float32:  # (iv) a PPO-like surrogate on these rows sends gradients into tc (non-zero and zero init)
            net.zero_grad(); (-(torch.tensor(np.random.default_rng(2).standard_normal(len(lp)), dtype=dt) * lpt).mean()).backward()
            print(f"(iv) {tag}: grad norms " + ", ".join(f"{n} {float(p.grad.norm()):.3e}" for n, p in net.named_parameters() if n.startswith("tc.")))

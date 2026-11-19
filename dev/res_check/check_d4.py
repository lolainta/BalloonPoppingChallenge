"""D4: gradient of the residual loss w.r.t. res_logstd at exactly ln 0.6 / ln 0.05 (raw parameter in async_ppo.res_gauss),
and the learner's post-step clamp keeps it inside the bounds. RL3_RES=0.25"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dev import async_ppo as ap  # noqa: E402
from dev.rl3_net import Net3  # noqa: E402
from BalloonPoppingGymEnv.agents.rl3_agent import RES_LS  # noqa: E402

torch.manual_seed(0)
o = torch.zeros(256, ap.LAUNCHED_COL + 1); o[:, ap.LAUNCHED_COL] = 1.0
mean = torch.zeros(256, 3)
for name, b, u_std in (("ln 0.6", RES_LS[1], 2.0), ("ln 0.05", RES_LS[0], 0.01)):  # samples wider / narrower: pushes outward
    net = Net3()
    with torch.no_grad():
        net.res_logstd.fill_(b)
    u = u_std * torch.randn(256, 3)
    lp, ent = ap.res_gauss(net, mean, u, o)
    loss = -(lp.mean()) - ap.ENT * ent.mean()  # maximise the log-prob of these samples (+ entropy bonus)
    opt = torch.optim.Adam([net.res_logstd], lr=0.1)
    opt.zero_grad(); loss.backward()
    g = net.res_logstd.grad.clone()
    old = net.res_logstd.detach().clone().requires_grad_(True)  # the previous clamp version, for contrast
    ls = old.clamp(*RES_LS); lpo = (-0.5 * ((u - mean) / ls.exp()) ** 2 - ls - 0.5 * ap.LOG2PI).sum(-1)
    (-(lpo.mean())).backward()
    opt.step(); raw = net.res_logstd.detach().clone()
    net.res_logstd.data.clamp_(*RES_LS)  # = the learner's line after opt.step()
    print(f"at {name} ({b:.4f}): grad {g.numpy().round(3)} (old clamp-in-loss grad {old.grad.numpy().round(3)}); "
          f"after step raw {raw.numpy().round(4)} -> clamped {net.res_logstd.detach().numpy().round(4)}; "
          f"inside bounds: {bool(((net.res_logstd >= RES_LS[0]) & (net.res_logstd <= RES_LS[1])).all())}")
    with torch.no_grad():  # and the gradient pointing back inward from the bound is followed
        net.res_logstd.fill_(b)
    lp, _ = ap.res_gauss(net, mean, (0.3 * torch.randn(256, 3)), o); net.res_logstd.grad = None; (-(lp.mean())).backward()
    print(f"   with samples of std 0.3 instead: grad {net.res_logstd.grad.numpy().round(3)} (moves back inside)")

"""Torch attention policy for RL2 (SB3 ActorCriticPolicy) + export to the numpy layout in agents/rl2_agent.py."""
import os

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy

from BalloonPoppingGymEnv.agents.rl2_agent import F, F_BASE, G_DIM, N_MAX, STRETCH

D, HEADS = int(os.environ.get("RL3_D", 64)), int(os.environ.get("RL3_HEADS", 4))  # §315: a widened checkpoint (dev/widen.py) sets both
LAYERS = int(os.environ.get("RL3_LAYERS", 2))  # §287: > 2 stacks zero-initialised (identity) blocks on a 2-layer checkpoint


def zero_blocks(net, start):
    """Blocks >= start become identities (their residual outputs are 0): the deeper net acts exactly like the shallower one."""
    with torch.no_grad():
        for b in net.blocks[start:]:
            for m in (b.att.out_proj, b.ff2):
                m.weight.zero_(); m.bias.zero_()


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.att = nn.MultiheadAttention(D, HEADS, batch_first=True)
        self.ff1, self.ff2 = nn.Linear(D, 2 * D), nn.Linear(2 * D, D)

    def forward(self, h, pad):
        z = self.ln1(h)
        h = h + self.att(z, z, z, key_padding_mask=pad, need_weights=False)[0]
        return h + self.ff2(torch.relu(self.ff1(self.ln2(h))))


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.g = nn.Linear(G_DIM, D)
        self.t0, self.t1 = nn.Linear(F_BASE, D), nn.Linear(D, D)  # (the chain columns: rl3_net.Net3.t0x, §288)
        self.blocks = nn.ModuleList([Block() for _ in range(LAYERS)])
        self.tgt = nn.Linear(D, 1)
        self.prior = nn.Parameter(torch.tensor([3.0]))  # logit += prior * (-t1/10): starts near "fastest target"
        self.str = nn.Linear(2 * D, len(STRETCH))
        self.v0, self.v1 = nn.Linear(2 * D, D), nn.Linear(D, 1)

    def forward(self, x):
        g, tok = x[:, :G_DIM], x[:, G_DIM:].reshape(-1, N_MAX, F)
        mask = tok[..., 14] > 0.5
        pad = ~mask
        pad = pad & mask.any(1, keepdim=True)  # all-padding rows: attend to everything instead of NaN
        ge = torch.relu(self.g(g))
        h = self.t1(torch.relu(self.t0(tok[..., :F_BASE]))) + ge[:, None]
        for b in self.blocks:
            h = b(h, pad)
        tl = self.tgt(h)[..., 0] + self.prior * (-tok[..., 6])
        tl = tl.masked_fill(~mask, -1e9)
        m = mask.float()[..., None]
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
        pg = torch.cat([pooled, ge], -1)
        return tl, self.str(pg), self.v1(torch.relu(self.v0(pg)))[:, 0]


class AttnPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, net_arch=[], **kwargs)
        self.net = Net()
        self.optimizer = self.optimizer_class(self.parameters(), lr=self._lr0, **self.optimizer_kwargs)

    def _build(self, lr_schedule):
        self._lr0 = lr_schedule(1)
        super()._build(lr_schedule)

    def _dist(self, obs):
        tl, sl, v = self.net(obs.float())
        return self.action_dist.proba_distribution(action_logits=torch.cat([tl, sl], -1)), v

    def forward(self, obs, deterministic=False):
        dist, v = self._dist(obs)
        a = dist.get_actions(deterministic=deterministic)
        return a, v[:, None], dist.log_prob(a)

    def evaluate_actions(self, obs, actions):
        dist, v = self._dist(obs)
        return v[:, None], dist.log_prob(actions), dist.entropy()

    def get_distribution(self, obs):
        return self._dist(obs)[0]

    def predict_values(self, obs):
        return self._dist(obs)[1][:, None]


def export(net, path):
    P = {"n_layers": len(net.blocks), "heads": HEADS}
    lin = lambda name, m: P.update({f"{name}.w": m.weight.detach().cpu().numpy().T, f"{name}.b": m.bias.detach().cpu().numpy()})
    lin("g", net.g); lin("t0", net.t0); lin("t1", net.t1); lin("tgt", net.tgt); lin("str", net.str); lin("v0", net.v0); lin("v1", net.v1)
    P["prior"] = net.prior.detach().cpu().numpy()
    for L, b in enumerate(net.blocks):
        q = f"l{L}."
        P[q + "ln1.w"], P[q + "ln1.b"] = b.ln1.weight.detach().cpu().numpy(), b.ln1.bias.detach().cpu().numpy()
        P[q + "ln2.w"], P[q + "ln2.b"] = b.ln2.weight.detach().cpu().numpy(), b.ln2.bias.detach().cpu().numpy()
        P[q + "in.w"], P[q + "in.b"] = b.att.in_proj_weight.detach().cpu().numpy().T, b.att.in_proj_bias.detach().cpu().numpy()
        P[q + "out.w"], P[q + "out.b"] = b.att.out_proj.weight.detach().cpu().numpy().T, b.att.out_proj.bias.detach().cpu().numpy()
        lin(q + "ff1", b.ff1); lin(q + "ff2", b.ff2)
    np.savez(path, **P)


if __name__ == "__main__":  # parity: torch vs numpy inference on real observations
    import sys
    sys.path.insert(0, ".")
    from BalloonPoppingGymEnv.agents.rl2_agent import RL2Agent
    torch.manual_seed(0)
    net = Net().eval()
    for p in net.parameters():
        p.data += 0.05 * torch.randn_like(p)
    export(net, "/tmp/rl2_parity.npz")
    X = np.load("/tmp/bc_test.npz")["X"][:20]
    with torch.no_grad():
        tl, sl, v = net(torch.tensor(X))
    ag = RL2Agent.__new__(RL2Agent)
    ag.net = dict(np.load("/tmp/rl2_parity.npz"))
    err = 0.0
    for i, x in enumerate(X):
        tn, sn, vn = ag.forward(x)
        m = x[G_DIM:].reshape(N_MAX, F)[:, 14] > 0.5
        err = max(err, np.abs(tn[m] - tl[i].numpy()[m]).max(), np.abs(sn - sl[i].numpy()).max(), abs(vn - v[i].item()))
    print(f"max |torch - numpy| = {err:.2e}")
    assert err < 1e-3

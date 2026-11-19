"""RL3 policy: RL2 attention net + launch head; phase masking makes the unused action parts deterministic
(pre-launch: target/stretch fixed; post-launch: launch fixed), so they add nothing to log-probs or gradients."""
import os

import numpy as np
import torch
import torch.nn as nn

from BalloonPoppingGymEnv.agents import rl2_agent as v2
from BalloonPoppingGymEnv.agents.rl3_agent import ATT, G_DIM, RES_LS
from dev import rl2_net
from dev.rl2_net import D, AttnPolicy, Net

NEG = -1e9


class Net3(Net):
    def __init__(self):
        super().__init__()
        self.g = nn.Linear(G_DIM, D)
        self.lau = nn.Linear(2 * D, 2)
        self.lprior = nn.Parameter(torch.tensor([1.0]))  # launch logit += lprior * (t - 20 s): starts near a 20 s launch
        self.att = None
        if os.environ.get("RL3_ATT"):  # learned launch attitude; starts at ~50% vertical (the old fixed choice)
            self.att = nn.Linear(2 * D, len(ATT))
            nn.init.zeros_(self.att.weight)
            with torch.no_grad():
                self.att.bias.zero_(); self.att.bias[0] = 3.0
        self.attc = None
        if os.environ.get("RL3_ATTC"):  # continuous launch attitude (Gaussian on pre-squash u, see rl3_agent.att_cont)
            self.attc = nn.Linear(2 * D, 2)
            nn.init.zeros_(self.attc.weight)
            with torch.no_grad():  # starts at ~80 deg, toward the balloons near the ground; std 1 in u
                self.attc.bias.copy_(torch.tensor([float(np.log(25.0 / 10.0)), 0.0]))
            self.attc_logstd = nn.Parameter(torch.zeros(2))
        self.res, self.res_bound = None, float(os.environ.get("RL3_RES") or 0)
        if self.res_bound:  # learned guidance residual (Gaussian on pre-tanh u, see rl3_agent.set_res); starts at exactly 0
            self.res = nn.Linear(2 * D, 3)
            nn.init.zeros_(self.res.weight); nn.init.zeros_(self.res.bias)
            self.res_logstd = nn.Parameter(torch.full((3,), float(np.log(0.3))))
        self.vl = None
        if os.environ.get("RL3_LAUNCH_SEP"):  # launch learned as a bandit: own baseline V_pad(pad state) of the episode's pops
            self.vl = nn.Sequential(nn.Linear(2 * D, D), nn.ReLU(), nn.Linear(D, 1))

    def forward(self, x, need_pg=False):
        g, tok = x[:, :G_DIM], x[:, G_DIM:].reshape(-1, v2.N_MAX, v2.F)
        mask = tok[..., 14] > 0.5
        pad = ~mask & mask.any(1, keepdim=True)
        ge = torch.relu(self.g(g))
        h = self.t1(torch.relu(self.t0(tok))) + ge[:, None]
        for b in self.blocks:
            h = b(h, pad)
        tl = (self.tgt(h)[..., 0] + self.prior * (-tok[..., 6])).masked_fill(~mask, NEG)
        m = mask.float()[..., None]
        pg = torch.cat([(h * m).sum(1) / m.sum(1).clamp(min=1), ge], -1)
        sl = self.str(pg)
        t_s = x[:, v2.G_DIM + 1] * 60.0
        ll = self.lau(pg) + torch.stack([torch.zeros_like(t_s), self.lprior * (t_s - 20.0)], -1)
        pre = (x[:, v2.G_DIM] < 0.5)[:, None]
        tl = torch.where(pre, torch.full_like(tl, NEG).index_fill(1, torch.tensor([0], device=tl.device), 0.0), tl)
        sl = torch.where(pre, torch.tensor([NEG, NEG, NEG, 0.0], device=sl.device).expand_as(sl), sl)
        ll = torch.where(pre, ll, torch.tensor([0.0, NEG], device=ll.device).expand_as(ll))
        out = [tl, sl, ll]
        if self.att is not None:  # post-launch: attitude fixed (index 0)
            al = self.att(pg)
            out.append(torch.where(pre, al, torch.full_like(al, NEG).index_fill(1, torch.tensor([0], device=al.device), 0.0)))
        if self.attc is not None:  # mean of u, appended after the categorical logits (only used on the pad)
            out.append(self.attc(pg))
        if self.res is not None:  # mean of the residual's u, last (only used after launch)
            out.append(self.res(pg))
        val = self.v1(torch.relu(self.v0(pg)))[:, 0]
        return (torch.cat(out, -1), val, pg) if need_pg else (torch.cat(out, -1), val)


class AttnPolicy3(AttnPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = Net3()
        self.optimizer = self.optimizer_class(self.parameters(), lr=self._lr0, **self.optimizer_kwargs)

    def _dist(self, obs):
        logits, v = self.net(obs.float())
        return self.action_dist.proba_distribution(action_logits=logits), v


def export(net, path):
    rl2_net.export(net, path)
    P = dict(np.load(path))
    P["lau.w"], P["lau.b"] = net.lau.weight.detach().cpu().numpy().T, net.lau.bias.detach().cpu().numpy()
    P["lprior"] = net.lprior.detach().cpu().numpy()
    if net.att is not None:
        P["att.w"], P["att.b"] = net.att.weight.detach().cpu().numpy().T, net.att.bias.detach().cpu().numpy()
    if net.attc is not None:
        P["attc.w"], P["attc.b"] = net.attc.weight.detach().cpu().numpy().T, net.attc.bias.detach().cpu().numpy()
        P["attc.logstd"] = net.attc_logstd.detach().cpu().numpy()
    if net.res is not None:  # the bound rides along, so evaluation / single files apply the residual as trained
        P["res.w"], P["res.b"] = net.res.weight.detach().cpu().numpy().T, net.res.bias.detach().cpu().numpy()
        P["res.logstd"] = net.res_logstd.detach().clamp(*RES_LS).cpu().numpy()
        P["res.bound"] = np.float32(net.res_bound)
    if os.environ.get("RL3_T_MIN"):  # record the launch floor this policy is trained with
        P["t_min"] = np.float32(os.environ["RL3_T_MIN"])
    for k, v in v2.parse_gnc_kw(os.environ.get("GNC_KW")).items():  # and the GNC tunables (agents apply them)
        P[f"gnc.{k}"] = np.float32(v)
    np.savez(path, **P)


def init_from_rl2(net3, rl2_state):
    """Copy an RL2 Net's weights; the global input layer gets zero columns for the 2 new features."""
    own = net3.state_dict()
    for k, v in rl2_state.items():
        if k == "g.weight":
            own[k].zero_(); own[k][:, : v.shape[1]] = v
        elif k in own and own[k].shape == v.shape:
            own[k].copy_(v)
    net3.load_state_dict(own)


def load_npz(net, path):
    """Inverse of export(): load exported numpy weights (e.g. an eval checkpoint) back into a Net3."""
    P = dict(np.load(path))
    sd = net.state_dict()
    def lin(name, key):
        sd[f"{key}.weight"].copy_(torch.tensor(P[f"{name}.w"].T))
        sd[f"{key}.bias"].copy_(torch.tensor(P[f"{name}.b"]))
    grow = net.att is not None and "att.w" in P and P["att.w"].shape[1] < net.att.out_features
    if grow:  # older checkpoint with fewer launch attitudes (ATT only grows at the end): keep the old ones, the new
        k = P["att.w"].shape[1]  # ones start state-independent at the old tilted options' mean logit (explored as often)
        w = np.zeros((P["att.w"].shape[0], net.att.out_features), np.float32); w[:, :k] = P["att.w"]
        b = np.full(net.att.out_features, P["att.b"][1:k].mean() if k > 1 else 0.0, np.float32); b[:k] = P["att.b"]
        P["att.w"], P["att.b"] = w, b
        print(f"launch attitudes {k} -> {net.att.out_features}", flush=True)
    for name in ("g", "t0", "t1", "tgt", "str", "v0", "v1", "lau") + (("att",) if net.att is not None and "att.w" in P else ()):
        lin(name, name)
    if net.attc is not None and "attc.w" in P:
        lin("attc", "attc"); sd["attc_logstd"].copy_(torch.tensor(P["attc.logstd"]))
    if net.res is not None and "res.w" in P:  # (an npz without it: the head stays at zero)
        lin("res", "res"); sd["res_logstd"].copy_(torch.tensor(P["res.logstd"]))
    sd["prior"].copy_(torch.tensor(P["prior"])); sd["lprior"].copy_(torch.tensor(P["lprior"]))
    for L in range(int(P["n_layers"])):
        q, k = f"l{L}.", f"blocks.{L}."
        for ln in ("ln1", "ln2"):
            sd[k + ln + ".weight"].copy_(torch.tensor(P[q + ln + ".w"])); sd[k + ln + ".bias"].copy_(torch.tensor(P[q + ln + ".b"]))
        sd[k + "att.in_proj_weight"].copy_(torch.tensor(P[q + "in.w"].T)); sd[k + "att.in_proj_bias"].copy_(torch.tensor(P[q + "in.b"]))
        sd[k + "att.out_proj.weight"].copy_(torch.tensor(P[q + "out.w"].T)); sd[k + "att.out_proj.bias"].copy_(torch.tensor(P[q + "out.b"]))
        lin(q + "ff1", k + "ff1"); lin(q + "ff2", k + "ff2")
    net.load_state_dict(sd)

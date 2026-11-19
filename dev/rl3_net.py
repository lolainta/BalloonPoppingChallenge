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


def res_logstd0(n):
    """Initial log std of an n-output res head: ln 0.3; u[5:8] (dv, RL3_RES3) ln RL3_RES_VSTD (0.1: dv = 10 tanh u -> ~1 m/s;
    0.3 -> ~3 m/s noise per axis cost ~2/3 of the pops of s4tc at the start, 0.1 ~5%, 2026-10-02)."""
    ls = np.full(n, np.log(0.3), np.float32)
    ls[5:] = np.log(float(os.environ.get("RL3_RES_VSTD", 0.1)))
    return ls

NEG = -1e9


class Net3(Net):
    def __init__(self):
        super().__init__()
        self.g = nn.Linear(G_DIM, D)
        self.lau = nn.Linear(2 * D, 2)
        self.lprior = nn.Parameter(torch.tensor([1.0]))  # launch logit += lprior * (t - 20 s): starts near a 20 s launch
        self.t0x = None
        if os.environ.get("RL3_CHAIN"):  # §288: the chain token columns 15..17, own input weights, zero = ignored at first
            self.t0x = nn.Linear(v2.F - v2.F_BASE, D, bias=False)
            nn.init.zeros_(self.t0x.weight)
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
            # RL3_RES2=1: 5 outputs, u[3] / u[4] scale the flown stretch / nav_N by exp(bound tanh u) (bounds RL3_RES_SB / _NB)
            # RL3_RES3=1: 8 outputs, u[5:8] -> arrival velocity offset vbound tanh u (m/s, bound RL3_RES_VB, gnc_agent._arrv)
            n = 8 if os.environ.get("RL3_RES3") else 5 if os.environ.get("RL3_RES2") else 3
            self.res_sbound, self.res_nbound = float(os.environ.get("RL3_RES_SB", 0.3)), float(os.environ.get("RL3_RES_NB", 0.15))
            self.res_vbound = float(os.environ.get("RL3_RES_VB", 10.0))
            self.res = nn.Linear(2 * D, n)
            nn.init.zeros_(self.res.weight); nn.init.zeros_(self.res.bias)
            self.res_logstd = nn.Parameter(torch.tensor(res_logstd0(n)))
        assert self.res is not None or not os.environ.get("RL3_RES2"), "RL3_RES2 needs RL3_RES"
        assert not os.environ.get("RL3_RES3") or os.environ.get("RL3_RES2"), "RL3_RES3 needs RL3_RES and RL3_RES2"
        self.tc = None
        if os.environ.get("RL3_TCOND"):  # target-conditioned speed gear / residual: the chosen target's final token embedding
            # adds tc(h_i) to the stretch logits and the residual mean (post-launch rows); starts at exactly 0
            self.tc = nn.Linear(D, 4 + (self.res.out_features if self.res is not None else 0))
            nn.init.zeros_(self.tc.weight); nn.init.zeros_(self.tc.bias)
        self.tc2 = None
        if os.environ.get("RL3_TCOND2"):  # + the same on the policy's second choice (rl3_agent.second), so the arrival velocity
            assert self.tc is not None, "RL3_TCOND2 needs RL3_TCOND"  # can aim at the next balloon; starts at exactly 0
            self.tc2 = nn.Linear(D, self.tc.out_features)
            nn.init.zeros_(self.tc2.weight); nn.init.zeros_(self.tc2.bias)
        self.fast, self.fast_k = None, int(os.environ.get("RL3_FAST") or 0)
        if self.fast_k:  # closed-loop residual (2026-10-05): on FAST rows (every fast_k sim steps between the decisions, target and
            # gear held) u = the decision's u + delta, delta ~ N(fast(pg) + fastc(h[current target]), fast_logstd); starts at exactly 0
            assert self.res is not None, "RL3_FAST needs RL3_RES"
            # RL3_FAST_DIMS=n (2026-10-06): a delta for u[0:n] only (u[n:] 0); RL3_FAST_STD=s: initial std (default 0.1), the
            # log std's lower clamp min(RES_LS[0], log s) (fast_lsmin, exported as "fast.lsmin" when lower)
            n = int(os.environ.get("RL3_FAST_DIMS") or self.res.out_features)
            self.fast, self.fastc = nn.Linear(2 * D, n), nn.Linear(D, n)
            for m in (self.fast, self.fastc):
                nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)
            ls0 = float(np.log(float(os.environ.get("RL3_FAST_STD") or 0.1)))
            self.fast_logstd = nn.Parameter(torch.full((n,), ls0))
            self.fast_lsmin = min(RES_LS[0], ls0)
        self.vl = None
        if os.environ.get("RL3_LAUNCH_SEP"):  # launch learned as a bandit: own baseline V_pad(pad state) of the episode's pops
            self.vl = nn.Sequential(nn.Linear(2 * D, D), nn.ReLU(), nn.Linear(D, 1))

    def forward(self, x, need_pg=False, tgt=None, detach=None):
        """tgt (B,): the chosen target per row (training: the sampled ACT[:, 0]) for a "tc" head; None = the argmax.
        detach (B,) bool (RL3_FAST: the FAST rows): their value is computed on detached trunk features (no gradient into the trunk)."""
        tok, mask, h, pg = self.trunk(x)
        tl = (self.tgt(h)[..., 0] + self.prior * (-tok[..., 6])).masked_fill(~mask, NEG)
        sl = self.str(pg)
        t_s = x[:, v2.G_DIM + 1] * 60.0
        ll = self.lau(pg) + torch.stack([torch.zeros_like(t_s), self.lprior * (t_s - 20.0)], -1)
        pre = (x[:, v2.G_DIM] < 0.5)[:, None]
        tc = tc2 = None
        if self.tc is not None:  # (pad rows: masked below / left out)
            ar = torch.arange(len(h), device=h.device)
            i = tl.argmax(1) if tgt is None else tgt.long()
            tc = self.tc(h[ar, i])
            sl = sl + tc[:, :4]
            if self.tc2 is not None:  # second choice j = argmax of tl over the live balloons other than i (an index: no gradient
                # through it); none (< 2 alive): 0. j is recomputed from the current weights, so pi(a|s) = p(i|s) p(stretch, u|s, i, j(s, i))
                # is a proper policy for every parameter value and the worker's log-prob (j from its own weights) is that of the
                # behaviour policy: the PPO ratio stays a valid importance ratio (j only switches where two logits tie)
                alive = mask.clone(); alive[ar, i] = False
                j = tl.masked_fill(~alive, NEG).argmax(1)
                tc2 = self.tc2(h[ar, j]) * alive.any(1, keepdim=True).to(h.dtype)
                sl = sl + tc2[:, :4]
        tl = torch.where(pre, torch.full_like(tl, NEG).index_fill(1, torch.tensor([0], device=tl.device), 0.0), tl)
        sl = torch.where(pre, torch.tensor([NEG, NEG, NEG, 0.0], device=sl.device).expand_as(sl), sl)
        ll = torch.where(pre, ll, torch.tensor([0.0, NEG], device=ll.device).expand_as(ll))
        out = [tl, sl, ll]
        if self.att is not None:  # post-launch: attitude fixed (index 0)
            al = self.att(pg)
            out.append(torch.where(pre, al, torch.full_like(al, NEG).index_fill(1, torch.tensor([0], device=al.device), 0.0)))
        if self.attc is not None:  # mean of u, appended after the categorical logits (only used on the pad)
            out.append(self.attc(pg))
        if self.fast is not None:  # mean of a FAST row's delta, before the residual's (only used on FAST rows)
            out.append(self.fast_mean(tok, mask, h, pg))
        if self.res is not None:  # mean of the residual's u, last (only used after launch)
            r = self.res(pg)
            out.append(r if tc is None else torch.where(pre, r, r + tc[:, 4:] if tc2 is None else r + tc[:, 4:] + tc2[:, 4:]))
        val = self.v1(torch.relu(self.v0(pg if detach is None else torch.where(detach[:, None], pg.detach(), pg))))[:, 0]
        return (torch.cat(out, -1), val, pg) if need_pg else (torch.cat(out, -1), val)

    def trunk(self, x):
        """-> tokens, alive mask, token features h, pooled + global features pg (shared by every head)."""
        g, tok = x[:, :G_DIM], x[:, G_DIM:].reshape(-1, v2.N_MAX, v2.F)
        mask = tok[..., 14] > 0.5
        pad = ~mask & mask.any(1, keepdim=True)
        ge = torch.relu(self.g(g))
        pre = self.t0(tok[..., :v2.F_BASE])
        if self.t0x is not None:
            pre = pre + self.t0x(tok[..., v2.F_BASE:])
        h = self.t1(torch.relu(pre)) + ge[:, None]
        for b in self.blocks:
            h = b(h, pad)
        m = mask.float()[..., None]
        return tok, mask, h, torch.cat([(h * m).sum(1) / m.sum(1).clamp(min=1), ge], -1)

    def fast_mean(self, tok, mask, h, pg):
        """RL3_FAST: a FAST row's delta mean, a readout of detached trunk features: the FAST rows never move the trunk that today's heads
        share (in mini runs that drift doubled the target entropy: dev/fast_check/ent_swap.py). The current target = the token flagged
        "is current target" (always one on FAST rows; none: the fastc term is 0)."""
        cur = (tok[..., 8] > 0.5) & mask
        hc = h[torch.arange(len(h), device=h.device), cur.float().argmax(1)]
        return self.fast(pg.detach()) + self.fastc(hc.detach()) * cur.any(1, keepdim=True).to(h.dtype)

    def fast_forward(self, x):
        """RL3_FAST, FAST rows only: (delta mean, value) = forward()'s with detach on every row, the trunk run without a graph (the loss of
        a FAST row reaches it only through detached features; building and back-propagating it anyway made the learner update ~13x
        slower, 2026-10-05)."""
        with torch.no_grad():
            tok, mask, h, pg = self.trunk(x)
        return self.fast_mean(tok, mask, h, pg), self.v1(torch.relu(self.v0(pg)))[:, 0]


class AttnPolicy3(AttnPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = Net3()
        assert self.net.tc is None, "RL3_TCOND needs the sampled target in the log-prob: dev/async_ppo.py only"
        self.optimizer = self.optimizer_class(self.parameters(), lr=self._lr0, **self.optimizer_kwargs)

    def _dist(self, obs):
        logits, v = self.net(obs.float())
        return self.action_dist.proba_distribution(action_logits=logits), v


def export(net, path):
    rl2_net.export(net, path)
    P = dict(np.load(path))
    P["lau.w"], P["lau.b"] = net.lau.weight.detach().cpu().numpy().T, net.lau.bias.detach().cpu().numpy()
    P["lprior"] = net.lprior.detach().cpu().numpy()
    if net.t0x is not None:  # §288: the agent computes the chain columns only for a policy that carries feat.chain
        P["t0x.w"], P["feat.chain"] = net.t0x.weight.detach().cpu().numpy().T, np.float32(1.0)
    if net.att is not None:
        P["att.w"], P["att.b"] = net.att.weight.detach().cpu().numpy().T, net.att.bias.detach().cpu().numpy()
    if net.attc is not None:
        P["attc.w"], P["attc.b"] = net.attc.weight.detach().cpu().numpy().T, net.attc.bias.detach().cpu().numpy()
        P["attc.logstd"] = net.attc_logstd.detach().cpu().numpy()
    if net.res is not None:  # the bound rides along, so evaluation / single files apply the residual as trained
        P["res.w"], P["res.b"] = net.res.weight.detach().cpu().numpy().T, net.res.bias.detach().cpu().numpy()
        P["res.logstd"] = net.res_logstd.detach().clamp(*RES_LS).cpu().numpy()
        P["res.bound"] = np.float32(net.res_bound)
        if net.res.out_features >= 5:
            P["res.sbound"], P["res.nbound"] = np.float32(net.res_sbound), np.float32(net.res_nbound)
        if net.res.out_features == 8:
            P["res.vbound"] = np.float32(net.res_vbound)
    if net.tc is not None:
        P["tc.w"], P["tc.b"] = net.tc.weight.detach().cpu().numpy().T, net.tc.bias.detach().cpu().numpy()
    if net.tc2 is not None:
        P["tc2.w"], P["tc2.b"] = net.tc2.weight.detach().cpu().numpy().T, net.tc2.bias.detach().cpu().numpy()
    if net.fast is not None:  # the cadence rides along (evaluation runs the FAST steps at it)
        for t in ("fast", "fastc"):
            P[f"{t}.w"], P[f"{t}.b"] = getattr(net, t).weight.detach().cpu().numpy().T, getattr(net, t).bias.detach().cpu().numpy()
        P["fast.logstd"] = net.fast_logstd.detach().clamp(net.fast_lsmin, RES_LS[1]).cpu().numpy()
        P["fast.k"] = np.int32(net.fast_k)
        if net.fast_lsmin < RES_LS[0]:  # (workers clamp the sampling std with it)
            P["fast.lsmin"] = np.float32(net.fast_lsmin)
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
        k, n = P["res.w"].shape[1], net.res.out_features
        if k < n:  # a 3- / 5-output head into a 5- / 8-output net: the new outputs start at zero (std res_logstd0)
            P["res.w"] = np.concatenate([P["res.w"], np.zeros((P["res.w"].shape[0], n - k), np.float32)], 1)
            P["res.b"] = np.concatenate([P["res.b"], np.zeros(n - k, np.float32)])
            P["res.logstd"] = np.concatenate([P["res.logstd"], res_logstd0(n)[k:]])
        lin("res", "res"); sd["res_logstd"].copy_(torch.tensor(P["res.logstd"]))
    for t in ("tc", "tc2"):  # (without it: zero); a 3- / 5-output residual's tc columns into a 5- / 8-output net: new ones 0
        if getattr(net, t) is not None and f"{t}.w" in P:
            k, n = P[f"{t}.w"].shape[1], getattr(net, t).out_features
            P[f"{t}.w"] = np.concatenate([P[f"{t}.w"], np.zeros((P[f"{t}.w"].shape[0], n - k), np.float32)], 1)
            P[f"{t}.b"] = np.concatenate([P[f"{t}.b"], np.zeros(n - k, np.float32)])
            lin(t, t)
    if net.fast is not None and "fast.w" in P:  # (without it: zero)
        lin("fast", "fast"); lin("fastc", "fastc"); sd["fast_logstd"].copy_(torch.tensor(P["fast.logstd"]))
    sd["prior"].copy_(torch.tensor(P["prior"])); sd["lprior"].copy_(torch.tensor(P["lprior"]))
    for L in range(int(P["n_layers"])):
        q, k = f"l{L}.", f"blocks.{L}."
        for ln in ("ln1", "ln2"):
            sd[k + ln + ".weight"].copy_(torch.tensor(P[q + ln + ".w"])); sd[k + ln + ".bias"].copy_(torch.tensor(P[q + ln + ".b"]))
        sd[k + "att.in_proj_weight"].copy_(torch.tensor(P[q + "in.w"].T)); sd[k + "att.in_proj_bias"].copy_(torch.tensor(P[q + "in.b"]))
        sd[k + "att.out_proj.weight"].copy_(torch.tensor(P[q + "out.w"].T)); sd[k + "att.out_proj.bias"].copy_(torch.tensor(P[q + "out.b"]))
        lin(q + "ff1", k + "ff1"); lin(q + "ff2", k + "ff2")
    if net.t0x is not None and "t0x.w" in P:
        sd["t0x.weight"].copy_(torch.tensor(P["t0x.w"].T))
    net.load_state_dict(sd)
    rl2_net.zero_blocks(net, int(P["n_layers"]))  # (a deeper net than the file: the extra blocks start as identities, §287)

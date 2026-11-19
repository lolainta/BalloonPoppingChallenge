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

SEG, MAX_LAG = 64, int(os.environ.get("RL3_MAX_LAG", 2))  # 3 for s40a (DECISION_LOG §253: lag-3 bursts dropped ~15%)
GAMMA, LAM, CLIP, ENT, VF, LR, EPOCHS, MAXGRAD = 0.99, 0.95, 0.1, 0.003, 0.5, 1e-4, 4, 0.5
GAMMA = float(os.environ.get("RL3_GAMMA", GAMMA))  # per-decision discount; used only by the learner (GAE, SIL returns)
# self-imitation (Oh et al. 2018): workers send whole episodes with >= SIL_MIN pops; the learner adds
# -log pi(a|s) * max(R - V(s), 0) on them (R = discounted return-to-go). Off unless SIL_MIN is set.
SIL_MIN = float(os.environ.get("SIL_MIN", "inf"))
SIL_W, SIL_VF, SIL_BUF, SIL_BATCH = float(os.environ.get("SIL_W", 1.0)), 0.01, int(os.environ.get("SIL_BUF", 300)), 1024
# SIL_Q = q (DECISION_LOG §297): of the demos past SIL_MIN, imitate only those whose pops reach the q-quantile of the recent
# episodes' pops (a fixed SIL_MIN stops selecting once the mean passes it: at 8.2 half of all episodes qualified)
SIL_Q = float(os.environ.get("SIL_Q", 0))
SIL_THR = float(os.environ.get("SIL_THR", 0))  # absolute pops threshold for the gate instead of the SIL_Q quantile (integer pops make quantiles sticky: 0.75 and 0.9 both gave 9, §312)
# RL3_TCOST = c (§297): learner-side shaping, reward - c per sim second of the decision (faster hops); the SIL filter,
# episode logs and best-episode records stay in pops
TCOST = float(os.environ.get("RL3_TCOST", 0))
# TEACH_BC = w > 0 (DECISION_LOG §201, expert iteration): plus w * behaviour cloning of the teacher demos (<run>/demos_in,
# dev/s4_teacher.py): -mean log pi(a_teacher | s) over a full SIL_BATCH of teacher rows, no advantage weighting (the SIL
# share above only learns where R - V > 0 and got half a batch: no measurable effect in s4teach, §197)
TEACH_BC = float(os.environ.get("TEACH_BC", 0.0))
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
from BalloonPoppingGymEnv.agents.rl2_agent import F as TOK_F  # noqa: E402  (18 since §288)
TOK0, TOK_CUR = LAUNCHED_COL + 2, 8  # rl3_agent.G_DIM (first token column), "is current target"
# the same for the NEXT target on hand-off rows (post-launch, the current target reached within RL3_HOENT_T s by t_hat):
# almost all next-target choices happen there (13 rows / episode, the argmax switches 76%), and the entropy over the
# alternatives had collapsed to ~0.1 nats (DECISION_LOG §158/§159). RL3_HOENT = target nats of that conditional entropy.
HOENT = float(os.environ.get("RL3_HOENT", 0))
HOENT_T = float(os.environ.get("RL3_HOENT_T", 1.2))
TOK_T, TOK_ALIVE = 9, 14  # token columns: t_hat / 10, alive flag


def hoent(tl, o, detach=True):
    """(mask, H_cond, p_switch) per row. tl = target logits (dead balloons already at NEG), o = observation rows.
    mask: launched and the current target (alive) has t_hat <= HOENT_T. H_cond: entropy over the alternatives (the
    current target's logit masked out). p_switch = 1 - softmax(tl)[current], detached: weighting by it spreads the
    choice among the alternatives without pushing keep-vs-switch."""
    import torch
    tok = o[:, TOK0:].reshape(len(o), -1, TOK_F)
    cur = (tok[:, :, TOK_CUR] > 0.5) & (tok[:, :, TOK_ALIVE] > 0.5)
    m = (o[:, LAUNCHED_COL] >= 0.5) & (cur & (tok[:, :, TOK_T] * 10.0 <= HOENT_T)).any(1)
    z = torch.log_softmax(tl.masked_fill(cur, -1e9), -1)
    ps = 1.0 - (torch.softmax(tl, -1) * cur).sum(-1)
    return m, -(z.exp() * z).sum(-1), (ps.detach() if detach else ps)


def gammas(obs, dt=None):
    """Per-step discount: 1 for decisions taken on the pad (RL3_WAIT_NODISCOUNT), else GAMMA; RL3_FAST: GAMMA ** (dt / 0.5 s), dt = the
    sim time to the next row (the horizon in time stays that of one GAMMA per 0.5 s decision)."""
    g = np.full(len(obs), GAMMA, np.float32) if not FAST else (GAMMA ** (np.asarray(dt, np.float64) / DT_DEC)).astype(np.float32)
    if WAIT_NODISCOUNT and len(obs) and np.asarray(obs).shape[1] > LAUNCHED_COL:
        g[np.asarray(obs)[:, LAUNCHED_COL] < 0.5] = 1.0
    return g


def lams(dt):
    """GAE lambda per row: LAM; RL3_FAST: LAM ** (dt / 0.5 s) (the advantages' time span stays that of the 0.5 s decisions)."""
    return None if not FAST else LAM ** (np.asarray(dt, np.float64) / DT_DEC)


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
NR = 5 if os.environ.get("RL3_RES2") else 3  # RL3_RES2: u (5,), u[3] / u[4] scale the flown stretch / nav_N (rl3_agent.set_res)
NR = 8 if os.environ.get("RL3_RES3") else NR  # RL3_RES3: u (8,), u[5:8] = the arrival velocity offset (gnc_agent._arrv)
TC2 = bool(os.environ.get("RL3_TCOND2"))
# closed-loop residual (RL3_FAST=k, 2026-10-05): between two decisions (FULL rows: every DECIDE steps / pop / target death) the env
# stops every k sim steps (FAST rows: target / gear held, u = the decision's u + delta, delta ~ the "fast" Gaussian head); one
# action column more, the "full" flag (1 / 0) before u: FAST rows count only the fast Gaussian (u columns = delta), FULL rows only
# the other heads. Learner only: workers take k from the npz (fast.k)
FAST = int(os.environ.get("RL3_FAST") or 0)
DT_DEC = 0.5  # s per FULL decision (rl2_agent.DECIDE sim steps of 0.01 s)
NF = int(os.environ.get("RL3_FAST_DIMS") or NR) if FAST else 0  # fast mean columns (before the residual's in the net output): a delta
# for u[0:NF] (RL3_FAST_DIMS, rl3_net.Net3; FAST rows store all NR u columns, u[NF:] 0)
# RL3_MB_SPLIT=k (§316): the FULL rows of a minibatch are forwarded in k pieces with gradient accumulation (every term's denominator is
# the whole minibatch's, so the pieces' losses add up to exactly the minibatch's loss): the widened s41a net (D 128, 8 heads, 6 layers)
# ran out of 24 GB of GPU memory on one 7200-row piece. RL3_MB_CHECK=1 prints the gradient difference vs the one-piece loss once.
MB_SPLIT = int(os.environ.get("RL3_MB_SPLIT", 1))
MB_CHECK = bool(os.environ.get("RL3_MB_CHECK"))


def fits(m, aw):
    """A worker message the learner can use: action width aw and, for an 8-output res head, stamped "nr" = 8 by a worker whose
    env applies u[5:8] (a worker on older code would apply u[3:8] as res / s_mul / n_mul at the same width); with a tc2 head
    (RL3_TCOND2) stamped "tc2" = 1 by a worker whose sampling applies it (older code ignores tc2: wrong log-probs, same width);
    with a fast head (RL3_FAST) stamped "fast" = 1 (FAST rows, "full" flag, "dt")."""
    return np.shape(m["act"])[1:] == (aw,) and (not RES or NR < 8 or m.get("nr") == NR) and (not TC2 or m.get("tc2") == 1) \
        and (not FAST or m.get("fast") == 1)


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


def attc_gauss(net, mean, u, o):
    """Log-prob / entropy of the Gaussian launch attitude u (RL3_ATTC), counted on the pad only (launched flag 0)."""
    ls = net.attc_logstd
    pre = (o[:, LAUNCHED_COL] < 0.5).float()
    lp = (-0.5 * ((u - mean) / ls.exp()) ** 2 - ls - 0.5 * LOG2PI).sum(-1)
    return lp * pre, (ls + 0.5 * (1.0 + LOG2PI)).sum() * pre


def fast_gauss(net, mean, A):
    """RL3_FAST: log-prob / entropy of a FAST row's delta (u columns) under the fast Gaussian (mean: the NF logits columns before the
    residual's, or Net3.fast_forward's), every row (only the FAST rows, "full" flag 0, are used)."""
    ls = net.fast_logstd  # (raw, clamped after each step, as res_logstd)
    flp = (-0.5 * ((A[:, -NR:][:, :mean.shape[1]] - mean) / ls.exp()) ** 2 - ls - 0.5 * LOG2PI).sum(-1)
    return flp, (ls + 0.5 * (1.0 + LOG2PI)).sum().expand_as(flp)


FAST_W = float(os.environ.get("RL3_FAST_W", 1.0))  # weight of the fast head's surrogate / entropy / SIL terms (each a mean over FAST rows)


def full_mask(A):
    """RL3_FAST: the rows where today's heads acted (FULL rows and pad rows: "full" flag 1)."""
    return A[:, -NR - 1] > 0.5


def new_mask(O, A):
    """RL3_NEWENT: float mask of the rows where the policy picks a NEW target (post-launch, no token flagged as the current target);
    RL3_FAST: FULL rows only."""
    m = ((O[:, LAUNCHED_COL] >= 0.5) & (O[:, TOK0:].reshape(len(O), -1, TOK_F)[:, :, TOK_CUR].max(1).values < 0.5)).float()
    return m * full_mask(A).float() if FAST else m


def flight_loss(net, O, A, OLP, ADV, RET, sizes, alphas, tg, launch=True, norm=None, fast_rows=False, nnew=None):
    """One minibatch's flight loss (PPO surrogate + value + entropy bonus + RL3_NEWENT / STRENT / HOENT temperature terms)
    -> (loss, temperature loss or None, {"newent" / "strent" / "hoent": (entropy, rows)}).
    RL3_FAST: every term of today's heads is a mean over the FULL (and pad) rows only, i.e. exactly this loss without RL3_FAST on
    those rows; the fast head's surrogate / entropy are means over the FAST rows (weight FAST_W); the value loss = the FULL rows' mean
    + the FAST rows' mean. Nothing from a FAST row reaches the shared trunk (the fast head and those rows' value read detached features,
    rl3_net.Net3.forward), so the trunk and today's heads get exactly the gradient they get without RL3_FAST from the same FULL rows.
    norm = (FULL rows, FAST rows, rows) of the whole minibatch when these rows are one piece of it (gradient accumulation), else these.
    fast_rows: these rows are all FAST rows -> the same loss (every other term weighs them 0) without the trunk's graph."""
    import torch
    if fast_rows:
        fmu, val = net.fast_forward(O)
        flp, fent = fast_gauss(net, fmu, A)
        fr = torch.exp(flp - OLP)
        fpg = -torch.min(fr * ADV, fr.clamp(1 - CLIP, 1 + CLIP) * ADV)
        nq = norm[1] if norm else len(O)
        return VF * (val - RET).pow(2).sum() / nq + FAST_W * (fpg.sum() / nq - ENT * fent.sum() / nq), None, {}
    log_alpha, log_alpha_s, log_alpha_h = alphas
    out = net(O, **tg(A), **({"detach": ~full_mask(A)} if FAST else {}))  # (RL3_FAST: FAST rows' value off the trunk)
    logits, val = (torch.cat(out[:2], -1), out[2]) if not launch else out
    lp, ent, k = 0.0, 0.0, 0
    for i, m in enumerate(sizes):
        z = torch.log_softmax(logits[:, k: k + m], -1); k += m
        lp = lp + z.gather(1, A[:, i: i + 1].long())[:, 0]
        hi = -(z.exp() * z).sum(-1)
        ent = ent + hi
        if i == 0:
            h_tgt = hi
        elif i == 1:
            h_str = hi
    if ATTC:  # Gaussian launch attitude, only on the pad (it is ignored after launch)
        glp, gent = attc_gauss(net, logits[:, k: k + 2], A[:, len(sizes): len(sizes) + 2], O)
        lp, ent = lp + glp, ent + gent
    if RES:  # guidance residual: last NR logits columns = its mean, last NR action columns = u
        rlp, rent = res_gauss(net, logits[:, -NR:], A[:, -NR:], O)
        lp, ent = lp + rlp, ent + rent
    fm = 1.0
    if FAST:
        full = full_mask(A); fm = full.to(logits.dtype)
        lp = torch.where(full, lp, OLP)  # FAST rows: ratio 1, and they are left out below (no gradient into today's heads)
    ratio = torch.exp(lp - OLP)
    pgr = -torch.min(ratio * ADV, ratio.clamp(1 - CLIP, 1 + CLIP) * ADV)
    if not FAST:
        if LAUNCH_SEP:  # pad rows are trained by launch_loss(); the flight loss uses post-launch rows only
            fw = (O[:, LAUNCHED_COL] >= 0.5).float()
            pg, ent = (pgr * fw).sum() / fw.sum().clamp(min=1), ent * fw
        else:
            pg = pgr.mean()
        loss = pg + VF * (val - RET).pow(2).mean() - ENT * ent.mean()
    else:
        w = fm * (O[:, LAUNCHED_COL] >= 0.5).float() if LAUNCH_SEP else fm
        q = 1.0 - fm
        nf, nq, nr = norm or (fm.sum().clamp(min=1), q.sum().clamp(min=1), len(O))
        pg = (pgr * w).sum() / (w.sum().clamp(min=1) if LAUNCH_SEP else nf)  # (LAUNCH_SEP: all FULL rows are in one piece)
        flp, fent = fast_gauss(net, logits[:, -NR - NF: -NR], A)
        fr = torch.exp(torch.where(full, OLP, flp) - OLP)
        fpg = -torch.min(fr * ADV, fr.clamp(1 - CLIP, 1 + CLIP) * ADV)
        se = (val - RET).pow(2)  # value: FULL rows' mean (= without RL3_FAST) + FAST rows' mean (value head only, trunk detached)
        loss = (pg + VF * ((se * fm).sum() / nf + (se * q).sum() / nq) - ENT * (ent * w).sum() / nf
                + FAST_W * ((fpg * q).sum() / nq - ENT * (fent * q).sum() / nq))
    a_loss, logs = None, {}
    if NEWENT:
        newm = new_mask(O, A)
        if newm.sum() > 0:
            h_new = (h_tgt * newm).sum() / newm.sum()
            # nnew (RL3_MB_SPLIT): these rows are one piece of a minibatch with nnew new-target rows -> this piece's share of its
            # entropy term and temperature loss (the pieces add up to the minibatch's)
            share = newm.sum() / nnew if nnew else 1.0
            loss = loss - log_alpha.exp().detach() * h_new * share
            a_loss = log_alpha * (h_new.detach() - NEWENT) * share  # alpha grows while the entropy is below target
            logs["newent"] = (float(h_new.detach()), float(newm.sum()))
    if STRENT:
        postm = (O[:, LAUNCHED_COL] >= 0.5).float() * fm
        if postm.sum() > 0:
            h_s = (h_str * postm).sum() / postm.sum()
            loss = loss - log_alpha_s.exp().detach() * h_s
            a_s = log_alpha_s * (h_s.detach() - STRENT)
            a_loss = a_s if a_loss is None else a_loss + a_s
            logs["strent"] = (float(h_s.detach()), float(postm.sum()))
    if HOENT:
        hom, h_c, w = hoent(logits[:, :sizes[0]], O)
        w = w * hom * fm
        if w.sum() > 0:
            h_ho = (h_c * w).sum() / w.sum()  # p_switch-weighted conditional entropy of the hand-off rows
            loss = loss - log_alpha_h.exp().detach() * h_ho
            a_h = log_alpha_h * (h_ho.detach() - HOENT)
            a_loss = a_h if a_loss is None else a_loss + a_h
            logs["hoent"] = (float(h_ho.detach()), float(w.sum()))
    return loss, a_loss, logs


def sil_loss(net, Od, Ad, Rd, sizes, tg, fast=False, bc=False):
    """Self-imitation loss of one demo batch: -log pi(a|s) max(R - V, 0) + value part. fast (RL3_FAST): a batch of FAST rows, log pi =
    the fast Gaussian; else today's heads (a batch of FULL / pad rows: the same as without RL3_FAST). bc (TEACH_BC): plain
    behaviour cloning of the categorical choices (target, speed gear, launch), -mean log pi, no advantage weight, no value part."""
    import torch
    if fast:  # (value and delta mean on detached trunk features, no trunk graph)
        md, vd = net.fast_forward(Od)
        lpd = fast_gauss(net, md, Ad)[0]
    else:
        ld, vd = net(Od, **tg(Ad))
        lpd, k2 = 0.0, 0
        for i, m in enumerate(sizes):
            z = torch.log_softmax(ld[:, k2: k2 + m], -1); k2 += m
            lpd = lpd + z.gather(1, Ad[:, i: i + 1].long())[:, 0]
        if ATTC and not bc:  # bc: the teacher's continuous parts are the policy's own means: cloning them only shrinks the
            lpd = lpd + attc_gauss(net, ld[:, k2: k2 + 2], Ad[:, len(sizes): len(sizes) + 2], Od)[0]  # stds -> categorical only
        if RES and not bc:
            lpd = lpd + res_gauss(net, ld[:, -NR:], Ad[:, -NR:], Od)[0]
    gap = (Rd - vd).clamp(min=0, max=5)
    sw = (Od[:, LAUNCHED_COL] >= 0.5).float() if LAUNCH_SEP else torch.ones_like(gap)
    if bc:
        return -(lpd * sw).sum() / sw.sum().clamp(min=1)
    return -(lpd * gap.detach() * sw).sum() / sw.sum().clamp(min=1) + SIL_VF * 0.5 * gap.pow(2).mean()


def shaped(rew, dt):
    """RL3_TCOST: the rewards less TCOST per sim second (dt = each row's sim time to the next row)."""
    rew = np.asarray(rew, np.float32)
    return rew - TCOST * np.asarray(dt, np.float32) if TCOST and dt is not None else rew


def gae(s, boot, launch=True):
    """Advantages of one segment s (obs, rew, done, val, dt) on its behaviour values, bootstrap value boot."""
    v = np.append(s["val"], boot); d = s["done"].astype(np.float32); r = shaped(s["rew"], s.get("dt"))
    A = np.zeros(len(r), np.float32); g = 0.0; gam = gammas(s["obs"], s.get("dt")) if launch else np.full(len(r), GAMMA)
    lam = lams(s.get("dt"))
    for t in reversed(range(len(r))):
        nonterm = 1.0 - d[t]
        delta = r[t] + gam[t] * v[t + 1] * nonterm - v[t]
        g = delta + gam[t] * (LAM if lam is None else lam[t]) * nonterm * g
        A[t] = g
    return A


def has_att(P):
    """The policy (weights P) has the launch attitude head this run randomises (RL3_ATT_RAND): columns 3 (/ 4) are an attitude only
    then (without one, column 3 is the res u or, with a fast head, the "full" flag)."""
    return ("attc.w" if ATTC else "att.w") in P


# ---------------------------------------------------------------------------------------------- worker
def _log_softmax(z):
    z = z - z.max()
    return z - np.log(np.exp(z).sum())


def act(agent, x, launch, rng, fast=False):
    """fast: the env stopped at a fast tick (env.fast). With a "fast" head (RL3_FAST) every action carries the "full" flag before u:
    FAST row [0, 0, 0, (attitude 0), 0, delta, (0 ..)], delta ~ N(fast mean, fast std) (log-prob: that Gaussian only); else 1."""
    from BalloonPoppingGymEnv.agents import rl2_agent as v2
    from BalloonPoppingGymEnv.agents.rl3_agent import RES_LS
    fh = launch and "fast.w" in agent.net
    if fh and fast and x[v2.G_DIM] >= 0.5:
        v = agent.forward(x)[3]
        m, ls = agent.fast_u, np.clip(agent.net["fast.logstd"], agent.net.get("fast.lsmin", RES_LS[0]), RES_LS[1])
        d = (m + np.exp(ls) * rng.standard_normal(len(m))).astype(np.float32)
        lp = float((-0.5 * ((d - m) / np.exp(ls)) ** 2 - ls - 0.5 * LOG2PI).sum())
        nh = 3 + (2 if "attc.w" in agent.net else 1 if "att.w" in agent.net else 0)  # (a delta for u[0:n] only: u[n:] 0)
        return np.concatenate([np.zeros(nh + 1), d, np.zeros(len(agent.net["res.b"]) - len(d))]), lp, float(v)
    a, lp, v = _act(agent, x, launch, rng)
    if launch and "res.w" in agent.net:  # residual u ~ N(mean, std) after launch; on the pad 0 with log-prob 0
        u = np.zeros(len(agent.net["res.b"]), np.float32)
        if x[v2.G_DIM] >= 0.5:
            m, ls = agent.res_u, np.clip(agent.net["res.logstd"], *RES_LS)
            u = (m + np.exp(ls) * rng.standard_normal(len(m))).astype(np.float32)  # float32: as the learner sees it
            lp += float((-0.5 * ((u - m) / np.exp(ls)) ** 2 - ls - 0.5 * LOG2PI).sum())
        a = np.concatenate([a.astype(float), [1.0] if fh else [], u])
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
    else:
        tl, sl, v = agent.forward(x)
    a, logp = [], 0.0
    for z in (tl, sl):  # the target first; a "tc" head (RL3_TCOND) then conditions the stretch (and act()'s residual) on it,
        if a and launch:  # a "tc2" head (RL3_TCOND2) on the second choice from the same target logits
            z = agent.cond(a[0], z, tl)
        lp = _log_softmax(z); j = rng.choice(len(z), p=np.exp(lp))
        a.append(j); logp += float(lp[j])
    if launch:
        a += [0, 0.0, 0.0] if "attc.w" in agent.net else ([0, 0] if att else [0])
    return np.array(a, float if "attc.w" in agent.net else int), logp, float(v)


def worker_loop(run, launch, rank):
    import warnings, logging
    warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
    if launch:
        from dev.rl3_env import GuidanceEnv3 as E, res_cols
        from BalloonPoppingGymEnv.agents.rl3_agent import RL3Agent as A
    else:
        from dev.rl2_env import GuidanceEnv as E
        from BalloonPoppingGymEnv.agents.rl2_agent import RL2Agent as A
    run = Path(run)

    def connect():  # (re)connect to whatever learner is published; survives learner restarts
        while True:
            try:
                host, port = (run / "learner.addr").read_text().split()
                if host == socket.gethostname().split(".")[0]:  # same node: loopback (cp08 maps its own name to a public,
                    host = "127.0.0.1"                         # firewalled IP; other nodes resolve it internally) §287
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
    nr = lambda: res_cols(env.agent_kw) if launch else 0  # noqa: E731  (the res columns this worker's env applies, see fits)
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
                for k in ("bound", "sbound", "nbound", "vbound"):  # res_sbound / res_vbound present = a 5- / 8-output head
                    env.agent_kw.pop(f"res_{k}", None)
                    if f"res.{k}" in d:
                        env.agent_kw[f"res_{k}"] = float(d[f"res.{k}"])
                env.agent_kw["chain"] = int("feat.chain" in d)  # §288: the env's agent computes the features: chain columns
                env.agent_kw.pop("fast_k", None)  # the fast tick cadence of a fast head (RL3_FAST), else none
                if "fast.w" in d:
                    env.agent_kw["fast_k"] = int(d["fast.k"])

    while not live.exists():
        time.sleep(2)
    reload()
    x, info = env.reset(seed=int(rng.integers(2**31)))
    ep_r, ep = 0.0, {"obs": [], "act": [], "rew": [], "dt": []}
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
        seg = {k: [] for k in ("obs", "act", "logp", "val", "rew", "done", "dt")}
        v_seg = version
        for _ in range(SEG):
            a, lp, v = act(agent, x, launch, rng, env.fast)
            t_row = float(env.obs["simulation_time"])
            on_pad = launch and LAUNCH_SEP and x[LAUNCHED_COL] < 0.5
            if on_pad:  # a pad decision: kept for the launch update
                attv = 1.0
                if rand_att and has_att(agent.net):  # random-attitude episode: every pad row carries it (also a forced launch at T_MAX)
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
            dt = float(env.obs["simulation_time"]) - t_row  # sim time to the next row (RL3_FAST discount)
            if on_pad:
                pad["went"].append(float(env.ag.launched()))  # this row launched (chosen, or forced at T_MAX)
            if ldir and launch and x[LAUNCHED_COL] < 0.5 and env.ag.launched():
                lrow = (np.asarray(x, np.float32), np.asarray(a, np.float32), float(rand_att))
            for k, val in zip(seg, (x, a, lp, v, r, done, dt)):
                seg[k].append(val)
            ep_r += r
            if SIL_MIN < np.inf:
                ep["obs"].append(x); ep["act"].append(a); ep["rew"].append(r); ep["dt"].append(dt)
            if done:
                send(("episode", {"r": ep_r, "seed": inf.get("seed"), "last_pop": inf.get("last_pop"), "version": v_seg}))
                if LAUNCH_SEP and pad["obs"] and one_width(pad["act"]):
                    send(("launch", {**{k: np.array(v, np.float32) for k, v in pad.items()}, "R": float(ep_r), "version": v_seg, "nr": nr(), "tc2": 1, "fast": 1}))
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
                    send(("demo", {"obs": np.array(ep["obs"], np.float32), "act": np.array(ep["act"]), "rew": np.array(ep["rew"], np.float32),
                                   "dt": np.array(ep["dt"]), "nr": nr(), "tc2": 1, "fast": 1}))
                ep_r, ep = 0.0, {"obs": [], "act": [], "rew": [], "dt": []}
                x2, _ = env.reset()
            x = x2
        boot = 0.0 if seg["done"][-1] else act(agent, x, launch, rng, env.fast)[2]
        if one_width(seg["act"]):  # else dropped: it straddles a width-changing reload
            send(("segment", v_seg, {**{k: np.array(v) for k, v in seg.items()}, "nr": nr(), "tc2": 1, "fast": 1}, boot))


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
    assert not FAST or RES, "RL3_FAST needs RL3_RES"
    assert MB_SPLIT == 1 or (FAST and not (STRENT or HOENT or LAUNCH_SEP)), \
        "RL3_MB_SPLIT > 1: RL3_FAST runs without STRENT / HOENT / LAUNCH_SEP only (their terms are per-piece means)"
    from BalloonPoppingGymEnv.agents.rl3_agent import RES_LS
    net = Net().to(dev)
    newp = [p for n, p in net.named_parameters() if n.startswith("res")]  # RL3_RES head: its own param group
    tcp = [p for n, p in net.named_parameters() if n.startswith("tc.")]  # RL3_TCOND head: its own param group (after res)
    tc2p = [p for n, p in net.named_parameters() if n.startswith("tc2.")]  # RL3_TCOND2 head: its own param group (after tc)
    fastp = [p for n, p in net.named_parameters() if n.startswith("fast")]  # RL3_FAST head (fast, fastc, fast_logstd): last
    deep = lambda n: n.startswith("blocks.") and int(n.split(".")[1]) >= 2  # noqa: E731
    deepp = [p for n, p in net.named_parameters() if deep(n)]  # RL3_LAYERS > 2 (§287): blocks >= 2 in their own group, after fast
    chainp = [p for n, p in net.named_parameters() if n.startswith("t0x.")]  # RL3_CHAIN (§288): its own group, last
    opt = torch.optim.Adam([p for n, p in net.named_parameters() if not n.startswith(("res", "tc.", "tc2.", "fast", "t0x."))
                            and not deep(n)],
                           lr=a.lr)
    # a tc head reads the chosen target: the sampled one (action column 0) wherever the learner recomputes log-probs (tc2 the
    # second choice, which Net3.forward derives from it and the current target logits)
    tg = (lambda A: {"tgt": A[:, 0].long()}) if tcp else (lambda A: {})  # noqa: E731
    log_alpha = torch.tensor(float(np.log(0.02)), device=dev, requires_grad=True)  # RL3_NEWENT temperature
    log_alpha_s = torch.tensor(float(np.log(0.02)), device=dev, requires_grad=True)  # RL3_STRENT temperature
    log_alpha_h = torch.tensor(float(np.log(0.02)), device=dev, requires_grad=True)  # RL3_HOENT temperature
    aopt = torch.optim.Adam([log_alpha, log_alpha_s, log_alpha_h], lr=3e-3)

    gauss = lambda mean, u, o: attc_gauss(net, mean, u, o)  # noqa: E731
    steps, version = 0, 0
    ck = run / "async_last.pt"
    if newp and not ck.exists():
        opt.add_param_group({"params": newp})
    if tcp and not ck.exists():
        opt.add_param_group({"params": tcp})
    if tc2p and not ck.exists():
        opt.add_param_group({"params": tc2p})
    if fastp and not ck.exists():
        opt.add_param_group({"params": fastp})
    if deepp and not ck.exists():
        opt.add_param_group({"params": deepp})
    if chainp and not ck.exists():
        opt.add_param_group({"params": chainp})
    if ck.exists():
        s = torch.load(ck, map_location=dev)
        add_res = newp and not any(k.startswith("res") for k in s["net"])  # saved without the RL3_RES head
        add_tc = bool(tcp) and "tc.weight" not in s["net"]  # saved without the RL3_TCOND head
        grow = newp and not add_res and s["net"]["res.weight"].shape[0] < net.res.out_features  # 3 -> 5 -> 8 (RL3_RES2 / 3)
        add_tc2 = bool(tc2p) and "tc2.weight" not in s["net"]  # saved without the RL3_TCOND2 head
        assert not (add_res and tcp and not add_tc), "a tc checkpoint without a residual head: not supported"
        assert not (add_tc and tc2p and not add_tc2), "a tc2 checkpoint without a tc head: not supported"
        add_fast = bool(fastp) and "fast.weight" not in s["net"]  # saved without the RL3_FAST head
        add_deep = bool(deepp) and "blocks.2.ln1.weight" not in s["net"]  # a 2-layer checkpoint into a deeper net (§287)
        add_chain = bool(chainp) and "t0x.weight" not in s["net"]  # a checkpoint without the chain columns (§288)
        assert not (add_deep and add_chain), "deepening and adding the chain input at once: not supported"
        assert not (add_deep and add_fast), "deepening and adding the fast head at once: not supported"
        assert not (grow and fastp and not add_fast), "growing the residual of a fast checkpoint: not supported"
        if grow:  # old outputs copied, the new ones keep their init (zero weight / bias, std rl3_net.res_logstd0); old res / tc Adam state dropped
            k = s["net"]["res.weight"].shape[0]
            grown = (("res.weight", net.res.weight, k), ("res.bias", net.res.bias, k), ("res_logstd", net.res_logstd, k))
            if tcp and not add_tc:  # a tc head's rows for the new residual outputs: zero (stretch rows and old ones kept)
                grown += (("tc.weight", net.tc.weight, 4 + k), ("tc.bias", net.tc.bias, 4 + k))
            if tc2p and not add_tc2:  # (the same for a tc2 head)
                grown += (("tc2.weight", net.tc2.weight, 4 + k), ("tc2.bias", net.tc2.bias, 4 + k))
            for n_, t, m in grown:
                t2 = t.detach().clone(); t2[:m] = s["net"][n_]; s["net"][n_] = t2
            g0 = s["opt"]["param_groups"][0]
            s["opt"] = {"state": {i: v for i, v in s["opt"]["state"].items() if i in g0["params"]}, "param_groups": [g0]}
        added = add_res or add_tc or add_tc2 or add_fast or add_deep or add_chain
        r = net.load_state_dict(s["net"], strict=not added)
        assert not added or (not r.unexpected_keys and all(k.startswith(("res", "tc.", "tc2.", "fast", "t0x.")) or deep(k) for k in r.missing_keys)), r
        if newp and not (add_res or grow):
            opt.add_param_group({"params": newp})
        if tcp and not (add_tc or grow):
            opt.add_param_group({"params": tcp})
        if tc2p and not (add_tc2 or grow):
            opt.add_param_group({"params": tc2p})
        if fastp and not add_fast:
            opt.add_param_group({"params": fastp})
        if deepp and not add_deep:
            opt.add_param_group({"params": deepp})
        if chainp and not add_chain:
            opt.add_param_group({"params": chainp})
        opt.load_state_dict(s["opt"]); steps, version = s["steps"], s["version"]
        if add_res or grow:  # keep the optimizer state of the old parameters; the new head starts with a fresh one
            opt.add_param_group({"params": newp})
            print("res head added (zero init)" if add_res else f"res head expanded {k} -> {net.res.out_features} outputs (new rows zero)", flush=True)
        if grow and tcp and not add_tc:  # (its group after the res one, as on a fresh start)
            opt.add_param_group({"params": tcp})
            print(f"tc head expanded {4 + k} -> {net.tc.out_features} outputs (new rows zero)", flush=True)
        if add_tc:  # the same for the tc head (after the res group)
            opt.add_param_group({"params": tcp})
            print(f"tc head added (zero init, {net.tc.out_features} outputs)", flush=True)
        if add_tc2 or (grow and tc2p):  # tc2 last; on a warm start every other group keeps its optimizer state
            opt.add_param_group({"params": tc2p})
            print(f"tc2 head added (zero init, {net.tc2.out_features} outputs)" if add_tc2 else
                  f"tc2 head expanded {4 + k} -> {net.tc2.out_features} outputs (new rows zero)", flush=True)
        if add_fast:  # last; every other group keeps its optimizer state
            opt.add_param_group({"params": fastp})
            print(f"fast head added (zero init, {net.fast.out_features} outputs, every {net.fast_k} sim steps)", flush=True)
        if add_deep:  # last; the old groups keep their optimizer state, the new blocks start as identities
            from dev.rl2_net import zero_blocks
            zero_blocks(net, 2); opt.add_param_group({"params": deepp})
            print(f"deepened 2 -> {len(net.blocks)} attention blocks (new ones zero-initialised: identities)", flush=True)
        if add_chain:  # last (the new t0x starts at zero: the chain columns are ignored until RL finds them useful)
            opt.add_param_group({"params": chainp})
            print("chain token columns added (t0x zero-initialised)", flush=True)
        if s.get("log_alpha") is not None:
            log_alpha.data.fill_(float(s["log_alpha"]))
        if s.get("log_alpha_s") is not None:
            log_alpha_s.data.fill_(float(s["log_alpha_s"]))
        if s.get("log_alpha_h") is not None:
            log_alpha_h.data.fill_(float(s["log_alpha_h"]))
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
    # RL3_OLD_LR=m (2026-10-08): every group but the fast head's learns at m x lr; s4fast2's fast head was worth +0.075 but
    # the rest drifted ~0.07 below its fork point (DECISION_LOG §175 / §178)
    OLD_LR = float(os.environ.get("RL3_OLD_LR") or 1.0)
    if OLD_LR != 1.0:
        for g in opt.param_groups:
            if not any(p is q for p in g["params"] for q in fastp + deepp + chainp):  # (new blocks, §287: full lr as the fast head)
                g["lr"] = a.lr * OLD_LR
        print("param group lr: " + " ".join(f"{len(g['params'])}:{g['lr']:.2e}" for g in opt.param_groups), flush=True)
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
    # the 10.x address every compute node reaches (user 2026-10-22); a node's own name may map to a firewalled public IP (§287)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as u:
        u.connect(("10.0.0.1", 9)); ip = u.getsockname()[0]  # (UDP connect: no packet sent, just the route's source address)
    (run / "learner.addr").write_text(f"{ip if ip.startswith('10.') else socket.gethostname().split('.')[0]} {lis.address[1]}\n")
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
    aw = len(sizes) + ((2 if ATTC else 0) + (NR if RES else 0) + (1 if FAST else 0) if a.launch else 0)  # action width of this run's policy
    wide = lambda m: fits(m, aw)  # noqa: E731  (stale workers of a run whose heads changed: other widths / code)
    # every finished training episode's env seed (organisers' seed record, 2026-09-29); written once per update
    seedw = csv.writer(sf := open(run / "episode_seeds.csv", "a", newline=""))
    if sf.tell() == 0:
        seedw.writerow(["seed", "version", "pops", "last_pop", "time"])
    seed_rows = []
    hist = []  # (time, steps, episodes so far) for recent rates
    n_eps_total = 0
    demos, n_demo = [], 0  # own good episodes: [(obs, act, return-to-go)], newest last (RL3_FAST: their FULL / pad rows)
    fdemos = []  # RL3_FAST: the same episodes' FAST rows (SIL of the fast head)
    tdemos, tseen = [], set()  # teacher episodes dropped into <run>/demos_in/*.npz (dev/search_teacher.py)

    def to_demo(d):
        R = np.zeros(len(d["rew"]), np.float32); g = 0.0; gam = gammas(d["obs"], d.get("dt")); rew = shaped(d["rew"], d.get("dt"))
        for t in reversed(range(len(R))):
            g = float(rew[t]) + gam[t] * g; R[t] = g
        return (np.asarray(d["obs"], np.float32), np.asarray(d["act"]), R)

    rets = []  # recent episodes' pops (SIL_Q quantile)

    def sil_thr():  # SIL_THR absolute, else the SIL_Q quantile of recent pops (SIL_MIN until 200 episodes are in)
        return SIL_THR or (np.quantile(rets, SIL_Q) if len(rets) >= 200 else SIL_MIN)

    def sil_q(buf):  # SIL_Q / SIL_THR: own demos (4th item: pops) at or above the threshold; never empties the buffer
        if not (SIL_Q or SIL_THR):
            return buf
        thr = sil_thr()
        return [d for d in buf if len(d) < 4 or d[3] >= thr] or buf

    def pick(buf, n):
        lens = np.array([len(d[0]) for d in buf]); k = np.random.randint(0, lens.sum(), n)
        cum = np.cumsum(lens); e = np.searchsorted(cum, k, side="right"); i = k - (cum[e] - lens[e])
        return [buf[a][0][b] for a, b in zip(e, i)], [buf[a][1][b] for a, b in zip(e, i)], [buf[a][2][b] for a, b in zip(e, i)]

    def sil_batch(demos, tdemos):  # half own episodes, half teacher episodes (when both exist)
        parts = [pick(b, SIL_BATCH // 2 if demos and tdemos else SIL_BATCH) for b in (sil_q(demos), tdemos) if b]
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
        out, _, pgf = net(Ol, need_pg=True, **tg(Al)); vp = net.vl(pgf.detach())[:, 0]
        out = out[:, :-(NR + NF)] if RES else out  # the residual (and the fast mean) is not a pad decision
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
                    dm = to_demo(msg[1]); n_demo += 1; pops = float(np.sum(msg[1]["rew"]))
                    if FAST:  # FULL / pad rows for today's heads, FAST rows for the fast head (two SIL batches)
                        fl = dm[1][:, -NR - 1] > 0.5
                        dm, fd = tuple(z[fl] for z in dm), tuple(z[~fl] for z in dm)
                        if len(fd[0]):
                            fdemos.append(fd + (pops,)); fdemos = fdemos[-SIL_BUF:]
                    demos.append(dm + (pops,)); demos = demos[-SIL_BUF:]
                    continue
                if msg[0] == "launch":
                    if wide(msg[1]):
                        lbuf.append(msg[1])
                    continue
                if msg[0] == "episode":
                    eps.append(msg[1]); eps = eps[-100:]; n_eps_total += 1; rets.append(float(msg[1]["r"])); rets = rets[-2000:]
                    ep = msg[1]
                    seed_rows.append([ep["seed"], ep["version"], ep["r"], ep["last_pop"], int(time.time())])
                    best.check(steps, [{"episode": {"r": ep["r"]}, "seed": ep["seed"], "last_pop": ep["last_pop"]}], net)
                else:
                    lags.append(max(version - msg[1], 0))
                    if version - MAX_LAG <= msg[1] <= version and wide(msg[2]):  # > version: acted with a pre-restart policy newer than the resumed checkpoint
                        segs.append(msg)  # RL3_FAST: --batch counts FULL / pad rows (the same sim time per update as without)
                        n += int((msg[2]["act"][:, -NR - 1] > 0.5).sum()) if FAST else SEG
                    else:
                        dropped += 1
            for fn in sorted((run / "demos_in").glob("*.npz")) if (run / "demos_in").exists() else []:
                if fn.name not in tseen and not fn.name.startswith("."):
                    tseen.add(fn.name)
                    try:
                        d = dict(np.load(fn))
                        if np.asarray(d["act"]).shape[1] != 3 + (2 if ATTC else 1 if os.environ.get("RL3_ATT") else 0) + NR * RES + (FAST > 0):
                            raise ValueError(f"action width {np.asarray(d['act']).shape[1]} does not match this run")
                        tdemos.append(to_demo(d)); tdemos = tdemos[-3000:]
                    except Exception as e:
                        print(f"bad teacher demo {fn.name}: {e!r}", flush=True)
            # GAE per segment on behaviour values
            obs, act_, oldlp, adv, ret = [], [], [], [], []
            for _, _, s, boot in segs:
                A = gae(s, boot, a.launch)
                obs.append(s["obs"]); act_.append(s["act"]); oldlp.append(s["logp"]); adv.append(A); ret.append(A + s["val"])
            T = lambda z, dt=torch.float32: torch.tensor(np.concatenate(z), dtype=dt, device=dev)
            O, ACT, OLP, ADV, RET = T(obs), T(act_), T(oldlp), T(adv), T(ret)  # float: continuous dims ride along
            if FAST:  # FULL / pad rows and FAST rows: advantages normalised per kind, every minibatch a quarter of each
                fullb = full_mask(ACT)
                for msk in (fullb, ~fullb):
                    if msk.sum() > 1:
                        ADV[msk] = (ADV[msk] - ADV[msk].mean()) / (ADV[msk].std() + 1e-8)
                fi, qi = torch.nonzero(fullb)[:, 0], torch.nonzero(~fullb)[:, 0]
                N, NQ = len(fi), len(qi)  # (N: FULL / pad rows, as without RL3_FAST)

                def minibatches():  # (FULL rows, FAST rows) per optimizer step
                    fch = fi[torch.randperm(N, device=dev)].split(max(N // 4, 1))
                    return list(zip(fch, torch.tensor_split(qi[torch.randperm(NQ, device=dev)], len(fch))))
            else:
                ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
                N, NQ = len(O), 0
                minibatches = lambda: torch.randperm(N, device=dev).split(N // 4)  # noqa: E731
            ent_log, sent_log, hent_log = [], [], []
            mb_check = MB_CHECK  # (the first minibatch of every update)
            for _ in range(EPOCHS):
                for j in minibatches():
                    opt.zero_grad()
                    if FAST:  # the FULL rows as one piece (= a minibatch without RL3_FAST), the FAST rows in pieces of that size
                        # (no trunk graph: fast_rows); gradients accumulate, denominators = the minibatch's
                        f_, q_ = j
                        nrm = (max(len(f_), 1), max(len(q_), 1), len(f_) + len(q_))
                        for p in q_.split(max(len(f_), 1)) if len(q_) else ():
                            flight_loss(net, O[p], ACT[p], OLP[p], ADV[p], RET[p], sizes, (log_alpha, log_alpha_s, log_alpha_h), tg,
                                        a.launch, nrm, fast_rows=True)[0].backward()
                        j = f_

                    def pieces(j, k):  # RL3_MB_SPLIT: the FULL rows' loss in k pieces -> (loss to back-propagate, a_loss, logs); every piece's
                        # terms carry the minibatch's denominators (nrm, nnew), so the pieces' losses add up to the one-piece loss; k > 1
                        # back-propagates each piece at once (its graph freed) and returns 0.0 as the loss
                        nnew = float(new_mask(O[j], ACT[j]).sum()) if k > 1 and NEWENT else None
                        loss, a_loss, logs = 0.0, None, {}
                        for p in (j.split(-(-len(j) // k)) if k > 1 else (j,)):
                            l_, al, lg = flight_loss(net, O[p], ACT[p], OLP[p], ADV[p], RET[p], sizes, (log_alpha, log_alpha_s, log_alpha_h), tg,
                                                     a.launch, nrm if FAST else None, nnew=nnew)
                            if k > 1:
                                l_.backward()
                            else:
                                loss = l_
                            if al is not None:
                                a_loss = al if a_loss is None else a_loss + al
                            for key, (h, n_) in lg.items():  # (entropy, rows) per head: rows-weighted mean over the pieces
                                h0, n0 = logs.get(key, (0.0, 0.0)); logs[key] = ((h0 * n0 + h * n_) / (n0 + n_), n0 + n_)
                        return loss, a_loss, logs

                    if mb_check:  # RL3_MB_CHECK: accumulated gradient vs the one-piece gradient on the same rows
                        mb_check = False
                        alphas = (log_alpha, log_alpha_s, log_alpha_h)
                        g_pre = [p.grad.detach().clone() if p.grad is not None else None for p in net.parameters()]
                        opt.zero_grad()
                        nnew = float(new_mask(O[j], ACT[j]).sum()) if NEWENT else None
                        lsum = 0.0
                        for p in j.split(-(-len(j) // MB_SPLIT)):
                            l_ = flight_loss(net, O[p], ACT[p], OLP[p], ADV[p], RET[p], sizes, alphas, tg, a.launch, nrm if FAST else None, nnew=nnew)[0]
                            l_.backward(); lsum += float(l_)
                        g_acc = [p.grad.detach().clone() if p.grad is not None else None for p in net.parameters()]
                        opt.zero_grad()
                        l1 = flight_loss(net, O[j], ACT[j], OLP[j], ADV[j], RET[j], sizes, alphas, tg, a.launch, nrm if FAST else None)[0]
                        l1.backward()
                        rows_ = []
                        for n_, ga, p in zip([n for n, _ in net.named_parameters()], g_acc, net.parameters()):
                            if ga is None and p.grad is None:
                                continue
                            g = p.grad if p.grad is not None else torch.zeros_like(p); ga = ga if ga is not None else torch.zeros_like(p)
                            rows_.append((float((ga - g).abs().max()), float(g.abs().max()), float(ga.abs().max()), n_))
                        rows_.sort(reverse=True)
                        print(f"mb_check RL3_MB_SPLIT={MB_SPLIT}: loss pieces {lsum:.6f} vs one piece {float(l1):.6f}; {len(j)} rows, nnew {nnew}; "
                              "worst |g_acc - g| (abs, |g|, |g_acc|, name): " + "; ".join(f"{d_:.2e} {g_:.2e} {ga_:.2e} {n_}" for d_, g_, ga_, n_ in rows_[:6]), flush=True)
                        opt.zero_grad()
                        for g, p in zip(g_pre, net.parameters()):  # (the FAST rows' gradients back in place)
                            if g is not None:
                                p.grad = g
                    loss, a_loss, logs = pieces(j, MB_SPLIT)
                    for key, lst in (("newent", ent_log), ("strent", sent_log), ("hoent", hent_log)):
                        if key in logs:
                            lst.append(logs[key])
                    if LAUNCH_SEP:
                        loss = loss + launch_loss(sum(sizes[:2]))
                    if demos or tdemos:  # (RL3_FAST: demos = the FULL / pad rows)
                        loss = loss + SIL_W * sil_loss(net, *sil_batch(demos, tdemos), sizes, tg)
                    if TEACH_BC and tdemos:  # expert iteration: clone the teacher's actions (§201)
                        loss = loss + TEACH_BC * sil_loss(net, *sil_batch([], tdemos), sizes, tg, bc=True)
                    if fdemos:  # RL3_FAST: the fast head's SIL on the demos' FAST rows
                        loss = loss + FAST_W * SIL_W * sil_loss(net, *sil_batch(fdemos, []), sizes, tg, fast=True)
                    if torch.is_tensor(loss):  # (RL3_MB_SPLIT > 1 without SIL terms: everything is back-propagated already)
                        loss.backward()
                    if FAST:  # clipped on its own: the fast head's gradient norm must not shrink the rest's step
                        torch.nn.utils.clip_grad_norm_(fastp, MAXGRAD)
                    torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if not any(p is q for q in fastp)] if FAST
                                                   else net.parameters(), MAXGRAD); opt.step()
                    if RES:
                        net.res_logstd.data.clamp_(*RES_LS)
                    if FAST:
                        net.fast_logstd.data.clamp_(net.fast_lsmin, RES_LS[1])
                    if a_loss is not None:
                        aopt.zero_grad(); a_loss.backward(); aopt.step()
                        with torch.no_grad():
                            log_alpha.clamp_(LOG_ALPHA_MIN, LOG_ALPHA_MAX); log_alpha_s.clamp_(LOG_ALPHA_MIN, LOG_ALPHA_MAX)
                            log_alpha_h.clamp_(LOG_ALPHA_MIN, LOG_ALPHA_MAX)
            steps += N; version += 1  # (RL3_FAST: FULL / pad rows only, so step counts stay comparable with other lines)
            lbuf = [d for d in lbuf if d["version"] >= version - 2 * MAX_LAG]
            publish()
            lag = np.mean([version - 1 - s[1] for s in segs])
            prog.writerow([steps, len(eps), f"{np.mean([e['r'] for e in eps]) if eps else float('nan'):.3f}", f"{time.time() - t0:.0f}"]); f.flush()
            seedw.writerows(seed_rows); sf.flush(); seed_rows = []
            print(f"v{version} steps {steps} batch {N} mean_r {np.mean([e['r'] for e in eps]) if eps else 0:.2f} lag {lag:.2f} "
                  + (f"launch_eps {len(lbuf)} " if LAUNCH_SEP else "") +
                  f"workers {len(conns)} sps {(steps - steps0) / max(time.time() - t0, 1):.0f} "
                  + (f"newent {np.average([e[0] for e in ent_log], weights=[e[1] for e in ent_log]) if ent_log else float('nan'):.3f} alpha {float(log_alpha.exp()):.4f} " if NEWENT else "")
                  + (f"strent {np.average([e[0] for e in sent_log], weights=[e[1] for e in sent_log]) if sent_log else float('nan'):.3f} alpha_s {float(log_alpha_s.exp()):.4f} " if STRENT else "")
                  + (f"hoent {np.average([e[0] for e in hent_log], weights=[e[1] for e in hent_log]) if hent_log else float('nan'):.3f} alpha_h {float(log_alpha_h.exp()):.4f} " if HOENT else "")
                  + (f"rows {N + NQ} fast_rows {NQ / max(N + NQ, 1):.2f} fast_std {float(net.fast_logstd.exp().mean()):.4f} " if FAST else "") +
                  (f"sil_q {len(sil_q(demos))}/{len(demos)}>={sil_thr():.1f} " if SIL_Q or SIL_THR else "") +
                  f"demos {n_demo}/{len(demos)} teacher {len(tdemos)} dropped {dropped}/{len(lags)} ({100 * dropped / max(len(lags), 1):.1f}%) lag_hist {np.bincount(np.minimum(lags, 5), minlength=6).tolist()}", flush=True)
            now = time.time(); hist.append((now, steps, n_eps_total)); hist = [h for h in hist if now - h[0] <= 300]
            span = max(now - hist[0][0], 1e-6)
            status = {"version": version, "steps": steps, "workers": len(conns), "lag": round(float(lag), 2),
                      "dropped_pct": round(100 * dropped / max(len(lags), 1), 1), "lag_hist": np.bincount(np.minimum(lags, 5), minlength=6).tolist(),
                      "dec_per_s": round((steps - hist[0][1]) / span, 1), "eps_per_h": round(3600 * (n_eps_total - hist[0][2]) / span),
                      "updated": int(now)}
            if FAST:  # (steps / batch / dec_per_s: FULL and pad rows; rows: all, FAST included)
                status.update(rows=N + NQ, fast_share=round(NQ / max(N + NQ, 1), 3))
            if NEWENT and ent_log:
                status.update(newent=round(float(np.average([e[0] for e in ent_log], weights=[e[1] for e in ent_log])), 4),
                              alpha=round(float(log_alpha.exp()), 5))
            if STRENT and sent_log:
                status.update(strent=round(float(np.average([e[0] for e in sent_log], weights=[e[1] for e in sent_log])), 4),
                              alpha_s=round(float(log_alpha_s.exp()), 5))
            if HOENT and hent_log:
                status.update(hoent=round(float(np.average([e[0] for e in hent_log], weights=[e[1] for e in hent_log])), 4),
                              alpha_h=round(float(log_alpha_h.exp()), 5))
            (run / "status.json").write_text(json.dumps(status))
            if time.time() - t_ck >= a.ckpt_every:
                t_ck = time.time()
                torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "steps": steps, "version": version,
                            "log_alpha": float(log_alpha.detach()) if NEWENT else None,
                            "log_alpha_s": float(log_alpha_s.detach()) if STRENT else None,
                            "log_alpha_h": float(log_alpha_h.detach()) if HOENT else None}, ck)
                export(net, run / f".{name}_{steps}.tmp.npz")  # atomic: a learner cancelled mid-export (DECISION_LOG §274,
                os.replace(run / f".{name}_{steps}.tmp.npz", run / f"{name}_{steps}.npz")  # s40b fork) left 41 of 77 arrays
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

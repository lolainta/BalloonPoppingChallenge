"""DP flight planner (dev/plans/dp_planner.md): receding-horizon beam-DP over balloon sequences on a balloon table.

Pure numpy, used by RL2Agent when dp > 0. A leg (node -> balloon m, tau) is the constant-force intercept that ZEM N=2
flies with t_go = tau, feasible iff |miss| <= a * w(tau - turn delay, burn left) (w: thrust until burnout, then coast).
Plan value (average-reward semi-MDP): sum over legs of p_hit - lam*tau - (1 - p_hit)*lam*dmiss, plus a greedy
one-leg tail. Labels with the same last balloon and similar arrival time / velocity are merged (keep the best).
"""
import numpy as np

DP_G = np.array([0.0, 0.0, -9.81])
DP_CFG = dict(D=4, W=24, B0=16, Bk=6, S0=(1.05, 1.25, 1.5), S1=(1.05, 1.3), lam=0.35, dmiss=2.0, mc=40, T=60,
              td=0.0, pc=(0.75, 0.90, 0.93, 20.0), dk=2, dv=3.0, per_m=2, post=1.0, alt=3.0, fz=1.0, hit=1.0,
              drag=None)


def dp_w(t, tb):
    """Displacement per unit specific force: held for t (t^2/2), or only for the burn left tb, then coasting."""
    return np.where(t <= tb, 0.5 * t * t, tb * (t - 0.5 * tb))


def dp_reach(B, dt, P, V, K0, tb, U, a, trate, td, T, mc, fz=1.0, hit=1.0, Ad=None):
    """First feasible leg time (table steps on 1..T, coarser for long legs) from nodes (P, V, thrust dir U, table index K0, burn left tb) to every
    balloon of the table B (K+1, M, 3) -> (S, M) int, 0 = unreachable. mc > 0: only the mc balloons nearest each node's
    ballistic position 1.5 s ahead are tested. fz: least vertical thrust (GNC never thrusts downward, _saturate);
    hit: miss distance that still pops (inside the balloon radius); Ad: per-node constant drag acceleration."""
    K, M = B.shape[0] - 1, B.shape[1]
    S = len(P)
    if mc and mc < M:
        k15 = np.minimum(K0 + int(round(1.5 / dt)), K)
        d = np.linalg.norm(B[k15] - (P + V * 1.5)[:, None], axis=-1)
        sel = np.argpartition(d, mc - 1, 1)[:, :mc]
    else:
        sel = np.broadcast_to(np.arange(M), (S, M))
    steps = np.r_[np.arange(1, 21), np.arange(22, 61, 2), np.arange(63, T + 1, 3)]  # 0.1 s to 2 s, 0.2 s to 6 s, then 0.3 s
    steps = steps[steps <= T]
    idx = K0[:, None] + steps[None]  # (S, T)
    inside = idx <= K
    tau = (steps * dt).astype(np.float32)[None, :, None, None]
    Bt = B[np.minimum(idx, K)[:, :, None], sel[:, None, :]]  # (S, T, mc, 3), float32 table
    acc = DP_G if Ad is None else (DP_G + Ad)[:, None, None]
    start = (P[:, None, None] + V[:, None, None] * tau + 0.5 * acc * tau ** 2).astype(np.float32)
    miss = Bt - start
    n = np.sqrt(np.einsum("stmk,stmk->stm", miss, miss))
    cos = np.einsum("stmk,sk->stm", miss, U.astype(np.float32)) / np.maximum(n, 1e-6)
    te = np.maximum(tau[..., 0] - td - np.arccos(np.clip(cos, -1.0, 1.0)) / trate, 0.0)  # thrust time after turning
    w = dp_w(te, tb.astype(np.float32)[:, None, None])
    ok = (n <= a * w + hit) & (miss[..., 2] >= fz * w - hit) & inside[:, :, None]
    r = np.where(ok.any(1), steps[np.argmax(ok, 1)], 0)
    out = np.zeros((S, M), int)
    np.put_along_axis(out, np.ascontiguousarray(sel), r, 1)
    return out


def dp_phit(s, tau, post, pc):
    """Prior hit probability of a leg flown at stretch s over tau seconds (half after burnout)."""
    return np.interp(s, (1.05, 1.25, 1.5), pc[:3]) * np.exp(-tau / pc[3]) * np.where(post, 0.5, 1.0)


def dp_legs(B, dt, P, V, K0, tb, U, si, m, k, a, trate, c, elev, Ad):
    """Fly legs (node si -> balloon m in k steps): feasibility incl. the altitude floor along the path and the
    post-burnout cap, arrival velocity and thrust direction (constant-force intercept) -> (ok, V', U', post)."""
    K = B.shape[0] - 1
    k = np.asarray(k, int)
    tau = k * dt
    tbn = tb[si]
    kk = np.minimum(K0[si] + k, K)
    miss = B[kk, m] - P[si] - V[si] * tau[:, None] - 0.5 * (DP_G + Ad[si]) * tau[:, None] ** 2
    nm = np.linalg.norm(miss, axis=1)
    cos = (miss * U[si]).sum(1) / np.maximum(nm, 1e-9)
    te = np.maximum(tau - c["td"] - np.arccos(np.clip(cos, -1.0, 1.0)) / trate, 0.0)
    we = dp_w(te, tbn)
    ok = (k >= 1) & (K0[si] + k <= K) & (nm <= a * we + c["hit"]) & (miss[:, 2] >= c["fz"] * we - c["hit"]) & (tau <= tbn + c["post"])
    wt = np.maximum(dp_w(tau, tbn), 1e-9)
    f = miss / wt[:, None]  # thrust (specific force beyond gravity)
    for q in (0.25, 0.5, 0.75):  # path altitude
        t = q * tau
        z = P[si, 2] + V[si, 2] * t + 0.5 * (Ad[si, 2] - 9.81) * t * t + f[:, 2] * dp_w(t, tbn)
        ok &= z >= elev + c["alt"]
    post = tau > tbn
    Vn = V[si] + (DP_G + Ad[si]) * tau[:, None] + f * np.minimum(tau, tbn)[:, None]
    th = np.where(post[:, None], Vn, f)  # burning: along the thrust; after burnout the nose follows the velocity
    Un = th / np.maximum(np.linalg.norm(th, axis=1, keepdims=True), 1e-9)
    return ok, Vn, Un, post


def dp_plan(B, dt, p, v, u, tb0, a, trate, elev, cfg=None, cur=-1, cur_k=0, force=()):
    """Plan from the rocket state (p, v, body z axis u, burn left tb0) over the balloon table B (K+1, M, 3).
    cur: current target (table index) and cur_k its committed arrival (steps, 0 = none); force: extra root balloons.
    -> dict: root legs (m, k, s, V, com = the committed leg) with the best plan value through each (val),
    the best plan's (m, k) sequence and expected pops (Ep), nodes expanded."""
    c = {**DP_CFG, **(cfg or {})}
    M, B32 = B.shape[1], B.astype(np.float32)
    lam, dmiss, pc = c["lam"], c["dmiss"], c["pc"]
    P, V, U = p[None].astype(float), v[None].astype(float), u[None].astype(float)
    K0, tb, J, Ep = np.zeros(1, int), np.array([float(tb0)]), np.zeros(1), np.zeros(1)  # Ep: expected pops of the legs
    used, first, seq = np.zeros((1, M), bool), np.zeros(1, int), np.zeros((1, 0, 2), int)
    best_val, best_seq, nodes, root = None, None, 0, None

    def drag(P, V):
        """Per-leg constant drag acceleration from each node's air-relative velocity (drag = (k rho, wind grid z, wind W))."""
        if c["drag"] is None:
            return np.zeros_like(V)
        kd, wz, W = c["drag"]
        vr = V - np.stack([np.interp(P[:, 2], wz, W[:, 0]), np.interp(P[:, 2], wz, W[:, 1]), 0.0 * P[:, 2]], 1)
        return -kd * np.linalg.norm(vr, axis=1, keepdims=True) * vr

    def tail(L, tbn):
        """Greedy one-leg tail: the value of the fastest next leg at the smallest stretch; nothing reachable: the burn
        left (up to the leg horizon) is charged at lam; 0 in the coast window."""
        Lf = np.where(L > 0, L, 10 ** 6).min(1) * dt
        s = c["S1"][0]
        tau = np.round(Lf / dt * s) * dt
        ph = dp_phit(s, tau, tau > tbn, pc)
        h = np.where(Lf < 10 ** 5, ph - lam * tau - (1 - ph) * lam * dmiss, -lam * np.minimum(tbn, c["T"] * dt))  # dead end
        return np.where(tbn <= 0, 0.0, h)

    def leaf(mask, L):
        """Leaves (final depth, or no feasible next leg): value = J + tail, best through each root leg."""
        nonlocal best_seq
        if not mask.any():
            return
        val = J[mask] + tail(L[mask], tb[mask])
        np.maximum.at(best_val, first[mask], val)
        i = int(np.argmax(val))
        if best_seq is None or val[i] > best_seq[0]:
            best_seq = (float(val[i]), seq[mask][i], float(Ep[mask][i]))

    for depth in range(c["D"] + 1):
        Ad = drag(P, V)
        L = dp_reach(B32, dt, P, V, K0, tb, U, a, trate, c["td"], c["T"], 0 if depth == 0 else c["mc"], c["fz"], c["hit"], Ad)
        L[used] = 0
        nodes += len(P)
        if depth == c["D"]:
            leaf(np.ones(len(P), bool), L)
            break
        # candidates: the nb fastest balloons of each node (+ forced roots), x stretches
        nb, S = (c["B0"], c["S0"]) if depth == 0 else (c["Bk"], c["S1"])
        Lc = np.where(L > 0, L, 10 ** 6)
        cand = np.argsort(Lc, 1, kind="stable")[:, :nb]
        si, ci = np.nonzero(Lc[np.arange(len(P))[:, None], cand] < 10 ** 6)
        m = cand[si, ci]
        if depth == 0:
            extra = [int(j) for j in set(force) | ({cur} if cur >= 0 else set()) if 0 <= j < M and L[0, j] > 0]
            prot = set(extra)
            m = np.unique(np.r_[m, extra]).astype(int)
            si = np.zeros(len(m), int)
        t1 = L[si, m]
        ns = len(S)
        si, m = np.repeat(si, ns), np.repeat(m, ns)
        s = np.tile(np.asarray(S, float), len(t1))
        k = np.maximum(np.round(np.repeat(t1, ns) * s), 1).astype(int)
        com = np.zeros(len(m), bool)
        if depth == 0 and cur >= 0 and cur_k > 0:  # the committed leg to the current target
            si, m, k = np.r_[si, 0], np.r_[m, cur], np.r_[k, cur_k]
            s = np.r_[s, cur_k / max(L[0, cur], 1)]
            com = np.r_[com, True]
        if not len(m):
            if depth:
                leaf(np.ones(len(P), bool), L)
            break
        ok, Vn, Un, post = dp_legs(B, dt, P, V, K0, tb, U, si, m, k, a, trate, c, elev, Ad)
        if depth:
            has = np.zeros(len(P), bool)
            has[si[ok]] = True
            leaf(~has, L)
        tau = k * dt
        ph = dp_phit(s, tau, post, pc)
        Jn = J[si] + ph - lam * tau - (1 - ph) * lam * dmiss
        Epn = Ep[si] + ph
        Jn = np.where(ok, Jn, -np.inf)
        # dominance merge: same last balloon, |dk| <= dk and |dv| <= dv -> keep the higher J; <= per_m labels a balloon
        cap = len(m) if depth == 0 else c["W"]
        keep, seen = [], {}
        kk_all = K0[si] + k
        for i in np.argsort(-Jn, kind="stable"):
            if not np.isfinite(Jn[i]):
                break
            lst = seen.setdefault(int(m[i]), [])
            if len(lst) >= c["per_m"] or any(abs(kk_all[i] - kk_all[j]) <= c["dk"] and
                                             np.abs(Vn[i] - Vn[j]).max() <= c["dv"] for j in lst):
                if not (depth == 0 and com[i]):  # the committed leg is always kept (compared by the agent)
                    continue
            lst.append(i)
            keep.append(i)
            if len(keep) >= cap:
                break
        if depth and prot:  # the best label through each protected first balloon (current target, forced) survives
            fm = root["m"][first[si]]
            for g in prot - set(fm[keep]):
                ci = np.flatnonzero((fm == g) & np.isfinite(Jn))
                if ci.size:
                    keep.append(int(ci[np.argmax(Jn[ci])]))
        if not keep:
            break
        keep = np.array(keep, int)
        if depth == 0:
            order = np.arange(len(keep))
            root = dict(m=m[keep], k=k[keep], s=s[keep], V=Vn[keep], com=com[keep])
            best_val = np.full(len(keep), -np.inf)
            first = order
        else:
            first = first[si[keep]]
        pk = si[keep]
        seq = np.concatenate([seq[pk], np.stack([m[keep], kk_all[keep]], 1)[:, None]], 1)
        used = used[pk].copy()
        used[np.arange(len(keep)), m[keep]] = True
        P, V, U = B[kk_all[keep], m[keep]].astype(float), Vn[keep], Un[keep]
        tb, K0, J, Ep = np.maximum(tb[pk] - k[keep] * dt, 0.0), kk_all[keep], Jn[keep], Epn[keep]
    if root is None:
        return None
    root["val"] = best_val
    root["seq"] = best_seq[1] if best_seq is not None else np.zeros((0, 2), int)
    root["Ep"] = best_seq[2] if best_seq is not None else 0.0
    root["nodes"] = nodes
    return root

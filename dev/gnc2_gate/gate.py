"""Plan-B gate (DECISION_LOG §132, dev/gnc2_diag/rl_alternatives_result.json synth rank 1): does next-target-aware GNC
add pops ON TOP OF the policy's own behaviour, in scenario 4, on the policy's own target order?

1. free : fly the RL3 policy normally (deterministic) on each seed -> launch, pops, own order (popped balloons in order).
2. teach: re-fly the seed with a teacher that follows that own order (the policy's own launch decision -> identical launch
   time / attitude; speed level = policy argmax at every decision, as dev/replay_teacher.py) under a variant:
     tau     : time-based blind handover (dev/gnc2_diag/verify_chaining/vhand.py): when the GNC min reach time to the
               current target is <= tau, it is released for good and the next alive balloon of the order becomes the
               target (a policy decision is forced so the speed level is re-picked, like the policy's own switch).
               None = strict (hold each target until it pops; replay_teacher SLACK=inf).
     keep    : r_ok of a must-hit blend on the released target (after a handover the old balloon stays a must-hit:
               a = (1-l) a_old + l a_new, l in 0..1 with the old still predicted to pass within r_ok m of its centre,
               min t_hit(old) + t_reach(new); none feasible -> pure a_new, the old is dropped).
     win     : chain_sim mode-2 blend while holding the current target (dev/gnc2_diag/hop_geometry/chain_sim.py): the
               next balloon of the order is revealed when the flown time-to-go (t_go * stretch) <= win (inf = whole hop).
     lead    : (T_LEAD, DELTA) terminal lead-out toward the next balloon of the order (dev/gnc2_diag/thrust_util/sim_lead.py)
     eps     : placebo: every burn step the force gets a relative eps kick in a random direction (rng seeded by pseed)
Nothing in the repo is modified (subclass of RL3Agent).

.venv/bin/python dev/gnc2_gate/gate.py run --seeds 34000-34191 [--task I --ntasks N] [--procs P] [--variants a,b07,..]
.venv/bin/python dev/gnc2_gate/gate.py report [out/*.jsonl]
"""
import argparse, contextlib, io, json, os, sys, time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).resolve().parent / "out"
CK = "runs/s4bw4/rl3_policy_25670400.npz"
SL = "runs/s4bw4/rl2_sl_25670400.npz"
SCEN = 4
G = np.array([0.0, 0.0, -9.81])
LAM = np.linspace(0.0, 1.0, 11)
TH = np.arange(0.0, 6.0, 0.02)
TOL = 1e-6
HZONE = 1.5  # (s) GNC min reach time below which the policy's switch away from a live target counts as a handover  # reach times live on a 0.1 s float grid (0.7000000000000001): compare with a tolerance

V = {"a": {}}
for tau in (0.5, 0.7, 0.9):
    V[f"b{int(tau * 10):02d}"] = dict(tau=tau)
for r in (0.8, 1.0, 1.2):
    V[f"c{int(r * 10):02d}"] = dict(tau=0.7, keep=r)
V["d"] = dict(tau=0.7, keep=1.0, win=1.1)
V["e"] = dict(tau=0.7, keep=1.0, win=np.inf)
V["d0"] = dict(win=1.1)  # pure chain_sim mode 2 on the strict teacher (no handover), for reference
V["e0"] = dict(win=np.inf)
for T in (0.5, 1.0, 1.5):
    for dl in (0.4, 0.7):
        V[f"L{int(T * 10):02d}_{int(dl * 10)}"] = dict(tau=0.7, lead=(T, dl))  # L05_7 = lead05, L10_7 = lead10
# ---- the TAU mimic fails the validity check (b07 - free ~ -0.7; blind switches miss balloons the policy's own switch
# does not): P = policy-triggered handover on the own order (at the policy's decision cadence, when its argmax leaves the
# current order target with min reach time <= HZONE, hand over to the next of the order); F = the free policy itself
# (its own targets/switches) with the next balloon = the one after its current target in the free run's own order.
V["Pb"] = dict(ptrig=HZONE)
for r in (0.8, 1.0, 1.2):
    V[f"Pc{int(r * 10):02d}"] = dict(ptrig=HZONE, keep=r)
V["Pd"] = dict(ptrig=HZONE, keep=1.0, win=1.1)
V["Pe"] = dict(ptrig=HZONE, keep=1.0, win=np.inf)
V["PL05_7"] = dict(ptrig=HZONE, lead=(0.5, 0.7))
V["PL10_7"] = dict(ptrig=HZONE, lead=(1.0, 0.7))
V["F0"] = dict(free=1)  # must reproduce the free run exactly
for r in (0.8, 1.0, 1.2):
    V[f"Fc{int(r * 10):02d}"] = dict(free=1, keep=r)
V["Fd"] = dict(free=1, keep=1.0, win=1.1)
V["Fe"] = dict(free=1, keep=1.0, win=np.inf)
V["Fd0"] = dict(free=1, win=1.1)
for T in (0.5, 1.0, 1.5):
    for dl in (0.4, 0.7):
        V[f"FL{int(T * 10):02d}_{int(dl * 10)}"] = dict(free=1, lead=(T, dl))
for k in range(1, 7):
    V[f"Fp{k}"] = dict(free=1, eps=1e-4 if k <= 3 else 1e-3, pseed=k)
for k in range(1, 7):  # placebos: p1-p3 relative force kick 1e-4, p4-p6 1e-3 (no systematic effect expected)
    V[f"p{k}"] = dict(tau=0.7, eps=1e-4 if k <= 3 else 1e-3, pseed=k)


def make_agents():
    from BalloonPoppingGymEnv.agents.gnc_agent import GNCAgent
    from BalloonPoppingGymEnv.agents.rl3_agent import RL3Agent

    class Free(RL3Agent):
        """The policy as is; logs the GNC min reach time to the old target when a decision switches away from it."""
        sw = None

        def decide(self, obs, action):
            old = self.cmd[0] if self.cmd is not None else None
            super().decide(obs, action)
            st = obs["balloon_status"][:, 0]
            if old is not None and st[old] == 1 and self.cmd is not None and self.cmd[0] != old and self.p_hat is not None:
                b = obs["balloon_states"][old]
                self.sw = (self.sw or []) + [float(self._reach_time(self.p_hat, self.v_hat, b[None, :3], b[None, 3:])[0])]

    class Gate(RL3Agent):
        def __init__(self, given, order, tau=None, keep=None, win=None, lead=None, eps=0.0, pseed=0, ptrig=None, free=0,
                     **kw):
            super().__init__(given, **kw)
            self.order, self.tau, self.keep, self.win, self.lead, self.eps = list(order), tau, keep, win, lead, eps
            self.ptrig, self.free = ptrig, free
            self.rng = np.random.default_rng(pseed)
            self.released, self.old, self.force = set(), None, False
            self._obs, self._tgo_now, self._pv = None, np.nan, None
            self.n = dict(hand=0, keep=0, win=0, lead=0)

        # ---- order bookkeeping
        def cur_of(self, st):
            return next((b for b in self.order if st[b] == 1 and b not in self.released), None)

        def next_of(self, st, cur):
            if cur not in self.order:
                return None
            return next((b for b in self.order[self.order.index(cur) + 1:] if st[b] == 1 and b not in self.released), None)

        def needs_decision(self, obs):
            return self.force or super().needs_decision(obs)

        def decide(self, obs, action):
            old = self.cmd[0] if self.cmd is not None else None
            super().decide(obs, action)
            self.force = False
            if self.free and self.keep and old is not None and self.cmd is not None and self.cmd[0] != old:
                st = obs["balloon_status"][:, 0]  # the policy switched away from a live target in the handover zone
                if st[old] == 1 and self._rt(self.p_hat, self.v_hat, obs, old) <= HZONE + TOL:
                    self.old = old; self.n["hand"] += 1

        def get_action(self, obs):
            if self.free:  # the policy flies itself (targets, speed, launch); only the GNC hooks below differ
                return RL3Agent.get_action(self, obs)
            t = float(obs["simulation_time"])
            if self.needs_launch_decision(obs):  # the policy's own launch (RL3Agent.get_action) -> same time / attitude
                _, _, ll, _, al = self.forward(self.features(obs))
                self.decide_launch(obs, int(np.argmax(ll)) == 1, al if "attc.w" in self.net else int(np.argmax(al)))
            elif not self.launched() and t >= 60.0:
                self.decide_launch(obs, True, getattr(self, "att_last", 0))
            self.pre_since += 1
            if self.p_hat is not None and self._burning(t) and self.needs_decision(obs):
                x = self.features(obs)
                tl, sl, *_ = self.forward(x)
                st = obs["balloon_status"][:, 0]
                b = self.cur_of(st)
                if (self.ptrig is not None and b is not None and self.cmd is not None and self.cmd[0] == b
                        and int(np.argmax(tl)) != b and self._rt(self.p_hat, self.v_hat, obs, b) <= self.ptrig + TOL):
                    nb = self.next_of(st, b)  # the policy itself would leave b now: hand over to the next of the order
                    if nb is not None:
                        self.released.add(b); self.n["hand"] += 1
                        self.old = b if self.keep else None
                        b = nb
                self.decide(obs, [b if b is not None else int(np.argmax(tl)), int(np.argmax(sl))])
            self.since += 1
            return GNCAgent.get_action(self, obs)

        def _rt(self, p, v, obs, b):
            s = obs["balloon_states"][b]
            return self._reach_time(p, v, s[None, :3], s[None, 3:])[0]

        # ---- GNC hooks
        def _select_target(self, p, v, obs):
            st = obs["balloon_status"][:, 0]
            self._obs, self._pv = obs, (p, v)
            if self.old is not None and st[self.old] != 1:
                self.old = None
            if self.free:
                b, tg = super()._select_target(p, v, obs)
                if b is not None and b == self.old:
                    self.old = None
                self._tgo_now = np.nan if tg is None else tg * self.stretch
                return b, tg
            cur = self.cur_of(st)
            if cur is None:  # order exhausted: the policy's pick (as replay_teacher)
                b, tg = super()._select_target(p, v, obs)
                self._tgo_now = np.nan if tg is None else tg * self.stretch
                return b, tg
            self.aim_prof = None
            self.stretch = self.cmd[1] if self.cmd is not None else 1.0
            t1 = self._rt(p, v, obs, cur)
            if self.tau is not None and t1 <= self.tau + TOL:
                nb = self.next_of(st, cur)
                t2 = self._rt(p, v, obs, nb) if nb is not None else np.inf
                if np.isfinite(t2):
                    self.released.add(cur); self.n["hand"] += 1; self.force = True
                    self.old = cur if self.keep else None
                    cur, t1 = nb, t2
            t1 = t1 if np.isfinite(t1) else self.t_grid[-1]
            self._tgo_now = t1 * self.stretch
            return cur, t1

        def _t_hit(self, p, v, net, bp, bv, horizon, r_ok):
            rel = (bp - p)[None] + (bv - v)[None] * TH[:, None] - 0.5 * net[None] * TH[:, None] ** 2
            ok = (np.linalg.norm(rel, axis=1) < r_ok) & (TH <= horizon)
            return TH[int(np.argmax(ok))] if ok.any() else np.inf

        def _blend(self, p, v, a_hit, hb, t_hit_go, a_nxt, nb, r_ok):
            """chain_sim mode 2: l in LAM, a = (1-l) a_hit + l a_nxt, must hit balloon state hb (pos, vel) within r_ok;
            -> list of (l, cost = t_hit + t_reach(next from the predicted arrival state), a) for the feasible l."""
            hor = 1.5 * max(t_hit_go * self.stretch, self.t_go_min) + 0.5
            out = []
            for lam in LAM:
                a = (1 - lam) * a_hit + lam * a_nxt
                net = self._saturate(a) + G
                th = self._t_hit(p, v, net, hb[:3], hb[3:], hor, r_ok)
                if not np.isfinite(th):
                    out.append((lam, np.inf, a)); continue
                ph, vh = p + v * th + 0.5 * net * th ** 2, v + net * th
                t2 = self._reach_time(ph, vh, (nb[:3] + nb[3:] * th)[None], nb[None, 3:])[0]
                out.append((lam, th + t2 if np.isfinite(t2) else np.inf, a, th))
            return out

        def _guidance(self, p, v, bpos, bvel, t_go):
            a_cur = super()._guidance(p, v, bpos, bvel, t_go)
            obs = self._obs
            if obs is None or self.target is None:
                return a_cur
            st = obs["balloon_status"][:, 0]
            bs = obs["balloon_states"]
            if self.old is not None:  # must-hit blend: old (released) balloon must still be hit, bend toward the new target
                ho = bs[self.old].astype(float)
                to = self._reach_time(p, v, ho[None, :3], ho[None, 3:])[0]
                if not np.isfinite(to):
                    self.old = None; return a_cur
                a_old = super()._guidance(p, v, ho[:3], ho[3:], to)
                cand = [c for c in self._blend(p, v, a_old, ho, to, a_cur, bs[self.target].astype(float), self.keep)
                        if len(c) > 3]  # old predicted hit (next may be unreachable: cost inf, still feasible)
                if not cand:
                    self.old = None; return a_cur
                lam, c, a, _ = min(cand, key=lambda c: (c[1], -c[0]))
                if lam < 1.0:
                    self.n["keep"] += 1
                return a
            if self.win is not None and self.target in self.order and t_go * self.stretch <= self.win + TOL:
                nb = self.next_of(st, self.target)
                if nb is None:
                    return a_cur
                bn = bs[nb].astype(float)
                tn = self._reach_time(p, v, bn[None, :3], bn[None, 3:])[0]
                if not np.isfinite(tn):
                    return a_cur
                a_next = super()._guidance(p, v, bn[:3], bn[3:], tn)
                res = self._blend(p, v, a_cur, np.concatenate([bpos, bvel]).astype(float), t_go, a_next, bn, 1.0)
                if len(res[0]) <= 3:  # even the current command is predicted to miss: leave it to the baseline
                    return a_cur
                c0 = res[0][1]
                best = min(res, key=lambda c: c[1])
                if np.isfinite(c0) and best[1] < c0 - 0.02 and best[0] > 0:
                    self.n["win"] += 1
                    return best[2]
            return a_cur

        def _spend_excess(self, f, v):
            f = super()._spend_excess(f, v)
            if self.eps:
                u = self.rng.normal(size=3)
                f = f + self.eps * np.linalg.norm(f) * u / np.linalg.norm(u)
            if self.lead is None or self.target is None or self._obs is None:
                return f
            T_LEAD, DELTA = self.lead
            tgo = self._tgo_now
            if not np.isfinite(tgo) or tgo > T_LEAD:
                return f
            obs = self._obs
            nb = self.next_of(obs["balloon_status"][:, 0], self.target)
            if nb is None:
                return f
            b, cur = obs["balloon_states"][nb], obs["balloon_states"][self.target]
            p_arr = cur[:3] + cur[3:] * tgo
            t2 = self._reach_time(p_arr, v, b[None, :3] + b[None, 3:] * tgo, b[None, 3:])[0]
            if not np.isfinite(t2):
                return f
            a_next = 2.0 * (b[:3] + b[3:] * (tgo + t2) - p_arr - v * t2) / t2 ** 2 - G
            d = a_next / max(np.linalg.norm(a_next), 1e-9)
            bb, c = f @ d, f @ f - self.a_thrust ** 2
            k = max(-bb + np.sqrt(max(bb * bb - c, 0.0)), 0.0)
            k = min(k, 2.0 * DELTA / max(tgo, 0.05) ** 2)
            if k > 0:
                self.n["lead"] += 1
            return f + k * d

    return Free, Gate


_ENV = {}


def fly(seed, variant=None, order=None):
    """-> record dict. variant None = free policy."""
    import warnings; warnings.filterwarnings("ignore")
    from dev import rl2_env
    Free, Gate = make_agents()
    if "env" not in _ENV:
        params, given = rl2_env.load_scenario_parameters(SCEN)
        _ENV.update(env=rl2_env.BalloonPoppingEnv(render_mode=None, parameters=params), given=given)
    env, given = _ENV["env"], _ENV["given"]
    t0 = time.time()
    try:
        obs, _ = rl2_env.reset_clean(env, seed)
    except Exception as e:  # some env seeds crash in reset (negative balloon mass)
        return dict(seed=seed, variant=variant or "free", error=repr(e))
    kw = dict(policy_path=str(ROOT / CK), sl_path=str(ROOT / SL))
    ag = Free(given, **kw) if variant is None else Gate(given, order, **V[variant], **kw)
    done, n, pops, R, P = False, 0, [], [], []
    while not done:
        obs, r, term, trunc, _ = env.step(ag.get_action(obs))
        done = term or trunc
        st = obs["balloon_status"][:, 0]
        k = int((st == 2).sum())
        if k > n:
            t = float(obs["simulation_time"])
            new = [int(b) for b in np.flatnonzero(st == 2) if b not in {q[1] for q in pops}]
            bs = obs["balloon_states"]
            rt = (ag._reach_time(ag.p_hat, ag.v_hat, bs[:, :3], bs[:, 3:]) if ag.p_hat is not None
                  else np.full(len(st), np.inf))
            for b in new:
                pops.append((t, b)); R.append(rt); P.append(None if ag.p_hat is None else ag.p_hat.copy())
            n = k
    burn_end = (ag.launched_at + ag.burn_time) if ag.launched_at is not None else np.inf
    # per hop (pop i -> pop i+1): Tmin = GNC min reach time to the next popped balloon at the moment of pop i
    tmin = [float(R[i][pops[i + 1][1]]) for i in range(len(pops) - 1)]
    bend = []
    for i in range(1, len(pops) - 1):
        if P[i - 1] is None or P[i + 1] is None:
            bend.append(None); continue
        u, w = P[i] - P[i - 1], P[i + 1] - P[i]
        bend.append(float(np.degrees(np.arccos(np.clip(u @ w / max(np.linalg.norm(u) * np.linalg.norm(w), 1e-9), -1, 1)))))
    rec = dict(seed=seed, variant=variant or "free", launch=float(ag.launch_time), burn_end=float(burn_end),
               incl_head=[float(x) for x in ag.incl_head], pops=[(round(t, 3), b) for t, b in pops],
               tmin=[x if np.isfinite(x) else None for x in tmin], bend=bend, secs=round(time.time() - t0, 1))
    if variant is None:
        rec["switch_tmin"] = [x if np.isfinite(x) else None for x in (ag.sw or [])]
    else:
        rec["n"] = ag.n
    return rec


def _fly(args):
    try:
        return fly(*args)
    except Exception as e:  # never lose a batch to one crash
        import traceback
        return dict(seed=args[0], variant=args[1] or "free", error=traceback.format_exc()[-800:])


def cmd_run(a):
    import multiprocessing as mp
    lo, hi = (int(x) for x in a.seeds.split("-"))
    seeds = list(range(lo, hi + 1))[a.task::a.ntasks]
    names = a.variants.split(",") if a.variants else list(V)
    OUT.mkdir(exist_ok=True)
    out = OUT / f"{a.tag}_{a.task:03d}.jsonl"
    done = set()
    if out.exists():  # resume
        for line in out.read_text().splitlines():
            r = json.loads(line); done.add((r["seed"], r["variant"]))
    fh = open(out, "a")
    free = {}
    if out.exists():
        for line in out.read_text().splitlines():
            r = json.loads(line)
            if r["variant"] == "free" and "error" not in r:
                free[r["seed"]] = r
    with mp.get_context("spawn").Pool(a.procs, maxtasksperchild=40) as pool:
        todo = [(s, None, None) for s in seeds if (s, "free") not in done]
        for r in pool.imap_unordered(_fly, todo):
            fh.write(json.dumps(r) + "\n"); fh.flush()
            if "error" not in r:
                free[r["seed"]] = r
            print(f"free {r['seed']} pops {len(r.get('pops', []))} {r.get('secs')}s {r.get('error', '')[:200]}", flush=True)
        todo = [(s, v, [b for _, b in free[s]["pops"]]) for v in names for s in seeds if s in free and (s, v) not in done]
        for i, r in enumerate(pool.imap_unordered(_fly, todo)):
            fh.write(json.dumps(r) + "\n"); fh.flush()
            if i % 20 == 0 or "error" in r:
                print(f"[{i + 1}/{len(todo)}] {r['variant']} {r['seed']} pops {len(r.get('pops', []))} {r.get('secs')}s "
                      f"{r.get('error', '')[:300]}", flush=True)


# ------------------------------------------------------------------ report
def load(files):
    recs = {}
    for f in files:
        for line in Path(f).read_text().splitlines():
            r = json.loads(line)
            if "error" not in r:
                recs[(r["seed"], r["variant"])] = r
    return recs


def stats(r):
    P = [t for t, _ in r["pops"]]
    inb = [t for t in P if t < r["burn_end"]]
    nb = len(inb)
    gaps = list(np.diff(inb))
    tm = [x for x in r["tmin"][: max(nb - 1, 0)] if x is not None]  # hops that start and end in the burn
    bd = [x for x in r["bend"][: max(nb - 2, 0)] if x is not None]
    return dict(pops=len(P), inburn=nb, post=len(P) - nb, gaps=gaps, tmin=tm, bend=bd,
                tmin_m=np.mean(np.minimum(tm, 8.0)) if tm else np.nan)  # Tmin is on a 0.1 s grid: means resolve better


def base_of(v):
    return "free" if v.startswith("F") or v == "free" else ("Pb" if v.startswith("P") else "b07")


def cmd_report(a):
    files = a.files or sorted(OUT.glob("gate*_*.jsonl"))
    recs = load(files)
    seeds = sorted({s for s, v in recs if v == "free"})
    names = ["free"] + [v for v in V if any((s, v) in recs for s in seeds)]
    S = {v: {s: stats(recs[(s, v)]) for s in seeds if (s, v) in recs} for v in names}
    fr = [recs[(s, "free")] for s in seeds]
    print(f"free runs {len(seeds)} seeds (errors: {sum(1 for f in files for l in Path(f).read_text().splitlines() if 'error' in json.loads(l))})")
    sw = [x for r in fr for x in r.get("switch_tmin", []) if x is not None]
    q = np.percentile(sw, [25, 50, 75])
    print(f"free policy switches away from a live target: {len(sw) / len(fr):.1f}/flight, GNC min reach time to the dropped "
          f"target median {q[1]:.2f} s (IQR {q[0]:.2f}-{q[2]:.2f}); {np.mean(np.array(sw) <= HZONE + TOL):.0%} within {HZONE} s")
    L = [abs(recs[(s, v)]["launch"] - recs[(s, "free")]["launch"]) for v in names for s in S[v]]
    print(f"launch time of every re-flight == free launch: max |diff| {max(L):.3f} s")
    if "F0" in S:
        same = [recs[(s, "F0")]["pops"] == recs[(s, "free")]["pops"] for s in S["F0"]]
        print(f"F0 (free policy through the F harness, no change) identical pops+times to free: {sum(same)}/{len(same)}")

    def diff(v, ref, key="pops"):
        ss = [s for s in S[v] if s in S[ref]]
        d = np.array([S[v][s][key] - S[ref][s][key] for s in ss], float)
        return d.mean(), d.std(ddof=1) / np.sqrt(len(d)), len(d)

    def med(v, k, ss=None):
        xs = [x for s in (ss or S[v]) for x in S[v][s][k]]
        return np.median(xs) if xs else np.nan

    def pmed(v, ref, k):  # medians over the seeds both have
        ss = [s for s in S[v] if s in S[ref]]
        return med(v, k, ss) - med(ref, k, ss)

    print("\nVALIDITY (mimic - free, paired, pops/seed; must be within 0.3):")
    for v in ("b05", "b07", "b09", "Pb"):
        if v in S:
            m, se, n = diff(v, "free")
            print(f"  {v:4} {m:+.3f} +- {se:.3f} (n={n})  {'OK' if abs(m) <= 0.3 else 'INVALID'}  {V[v]}")
    print(f"\n{'variant':8} {'n':>3} {'pops':>6} {'vs base':>18} {'vs free':>15} {'in-burn':>7} {'post':>5} {'gap':>5} "
          f"{'Tmin':>5} {'dTmin':>6} {'bend':>5}  spec")
    for v in names:
        b = base_of(v)
        n = len(S[v])
        pops = np.mean([S[v][s]["pops"] for s in S[v]])
        vb = "%s %+.3f+-%.3f" % ((b,) + diff(v, b)[:2]) if v != b else "-"
        vf = "%+.3f+-%.3f" % diff(v, "free")[:2] if v != "free" else "-"
        ib = np.mean([S[v][s]["inburn"] for s in S[v]]); po = np.mean([S[v][s]["post"] for s in S[v]])
        dt = pmed(v, b, "tmin") if v != b else 0.0
        print(f"{v:8} {n:3} {pops:6.3f} {vb:>18} {vf:>15} {ib:7.3f} {po:5.2f} {med(v, 'gaps'):5.2f} {med(v, 'tmin'):5.2f} "
              f"{dt:+6.2f} {med(v, 'bend'):5.1f}  {V.get(v, '')}")

    def tdiff(x, y):
        d = [S[x][s]["tmin_m"] - S[y][s]["tmin_m"] for s in S[x] if s in S[y]]
        return np.nanmean(d)

    def line(x, y):
        if x not in S or y not in S:
            return
        m, se, n = diff(x, y); mi, sei, _ = diff(x, y, "inburn"); mp, sep, _ = diff(x, y, "post")
        print(f"  {x}-{y}: {m:+.3f} +- {se:.3f} (n={n})  in-burn {mi:+.3f} +- {sei:.3f}  post {mp:+.3f} +- {sep:.3f}  "
              f"median Tmin {pmed(x, y, 'tmin'):+.2f} s (mean/seed {tdiff(x, y):+.3f} s)  gap {pmed(x, y, 'gaps'):+.2f} s  bend {pmed(x, y, 'bend'):+.1f} deg")
    print("\nKEY CONTRASTS (pops/seed, paired SE):")
    for x, y in [("Fd", "free"), ("Fd0", "free"), ("Fc08", "free"), ("Fc10", "free"), ("Fc12", "free"), ("Fe", "Fd"),
                 ("Fe", "free"), ("Fd", "Fc10"), ("FL05_7", "free"), ("FL10_7", "free"),
                 ("Pd", "Pb"), ("Pc10", "Pb"), ("Pe", "Pd"), ("PL05_7", "Pb"), ("PL10_7", "Pb"),
                 ("a", "b07"), ("d", "b07"), ("c10", "b07")]:
        line(x, y)
    for fam, base, pl, le in (("F", "free", "Fp", "FL"), ("TAU", "b07", "p", "L")):
        pls = [v for v in names if v.startswith(pl) and v[len(pl):].isdigit()]
        les = [v for v in names if v.startswith(le) and "_" in v]
        if len(pls) < 2:
            continue
        pd = [diff(v, base)[0] for v in pls]
        print(f"\n{fam} placebos vs {base}: {' '.join(f'{v} {x:+.3f}' for v, x in zip(pls, pd))}; mean {np.mean(pd):+.3f} "
              f"sd {np.std(pd, ddof=1):.3f} max {max(pd):+.3f} (best-of-{len(pls)} inflation)")
        ch = [np.mean([recs[(s, v)]["pops"] != recs[(s, base)]["pops"] for s in S[v] if s in S[base]]) for v in pls]
        print(f"  fraction of seeds whose pop list differs from {base}: {' '.join(f'{x:.2f}' for x in ch)}")
        if les:
            ld = {v: diff(v, base)[0] for v in les}
            bl = max(ld, key=ld.get)
            print(f"{fam} lead settings vs {base}: {' '.join(f'{v} {x:+.3f}' for v, x in ld.items())}")
            print(f"  best lead {bl} {ld[bl]:+.3f} (SE {diff(bl, base)[1]:.3f}); best lead - best placebo {ld[bl] - max(pd):+.3f}; "
                  f"best lead - placebo mean {ld[bl] - np.mean(pd):+.3f}")
    # F family: how often the next popped balloon is the oracle successor (the next balloon the GNC was shaped toward)
    fo = {s: [b for _, b in recs[(s, "free")]["pops"]] for s in seeds}
    for v in [v for v in names if v.startswith("F") and v != "F0"]:
        hit = tot = 0
        for s in S[v]:
            o = fo[s]; nx = {o[i]: o[i + 1] for i in range(len(o) - 1)}
            pp = [b for _, b in recs[(s, v)]["pops"]]
            for i in range(len(pp) - 1):
                if pp[i] in nx:
                    tot += 1; hit += nx[pp[i]] == pp[i + 1]
        print(f"  {v}: next pop == own-order successor in {hit}/{tot} hops ({hit / max(tot, 1):.0%})", end="")
    print()
    if "Fd" in S:
        m, se, n = diff("Fd", "free")
        print(f"\nDECISION INPUT (F family): (d)-(b) = {m:+.3f} +- {se:.3f} pops/seed (n={n}); median hop-start Tmin (d)-(b) "
              f"= {pmed('Fd', 'free', 'tmin'):+.2f} s, mean per seed {tdiff('Fd', 'free'):+.3f} s")
    if "Pd" in S:
        m, se, n = diff("Pd", "Pb")
        print(f"DECISION INPUT (P family): (d)-(b) = {m:+.3f} +- {se:.3f} pops/seed (n={n}); median hop-start Tmin (d)-(b) "
              f"= {pmed('Pd', 'Pb', 'tmin'):+.2f} s, mean per seed {tdiff('Pd', 'Pb'):+.3f} s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("--seeds", default="34000-34191"); r.add_argument("--task", type=int, default=0)
    r.add_argument("--ntasks", type=int, default=1); r.add_argument("--procs", type=int, default=8)
    r.add_argument("--variants", default=""); r.add_argument("--tag", default="gate")
    p = sp.add_parser("report"); p.add_argument("files", nargs="*")
    a = ap.parse_args()
    cmd_run(a) if a.cmd == "run" else cmd_report(a)

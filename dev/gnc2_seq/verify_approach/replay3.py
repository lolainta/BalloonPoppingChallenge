"""Linear replay of a route (list of per-leg options) with instrumentation of the offset state at every pop.
usage: replay.py MODE SEED OPT0,OPT1,...   (MODE appr|chaos; opts beyond list = 0; 'cyc' = cycle 1..7)"""
import sys, time, json, warnings, logging
import numpy as np
sys.path.insert(0, "./dev/gnc2_seq/approach")
import approach_fork as af
warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
mode, seed, spec = sys.argv[1], int(sys.argv[2]), sys.argv[3]
TSW = float(sys.argv[4]) if len(sys.argv) > 4 else 0.8
Base = af.make_teacher_cls()
class Inst(Base):
    last = None
    hist = []
    def _guidance(self, p, v, bpos, bvel, t_go):
        o = self.off
        if o is not None and self.target == o["tgt"]:
            self.last = (max(t_go * self.stretch, self.t_go_min), o["on"])
        r = super()._guidance(p, v, bpos, bvel, t_go)
        if o is not None and self.target == o["tgt"]:
            self.last = self.last + (o["on"],)
            self.hist.append((round(self.t_now, 2), round(self.last[0], 3), self.last[2]))
        return r
env, obs, given = af.sd.make_env(seed)
ag = Inst(given, **af.KW); ag.rec = []
ag.follow = False
def step():
    global obs
    env.unwrapped._start_wall_time = time.monotonic()
    ag.t_now = float(obs["simulation_time"])
    a = ag.get_action(obs)
    TR.append((round(ag.t_now,2), ag.target, None if ag.cmd is None else ag.cmd[0], getattr(ag, "eob_f", None) is not None, bool(getattr(ag,"plan_on",False)), None if ag.off is None else ag.off["on"], ag.p_hat.tolist() if ag.p_hat is not None else None))
    obs, r, te, tr, _ = env.step(a); return te or tr
done = False
TR = []
while not done and not af.sd.at_decision(ag, obs):
    done = step()
ag.follow = True
pops, events, leg = [], [], 0
def opt_for(i):
    if spec == "cyc": return 1 + i % 7
    l = [int(x) for x in spec.split(",") if x != ""]
    return l[i] if i < len(l) else 0
while not done and ag._burning(float(obs["simulation_time"])):
    tl, sl, *_ = ag.forward(ag.features(obs))
    order = [int(i) for i in np.argsort(-tl)[:2] if tl[i] > -1e8]
    if not order: break
    s = int(np.argmax(sl)); tgt = order[0]; second = order[1] if len(order) > 1 else None
    j = opt_for(leg); ag.force = (tgt, s); ag.off = None
    if mode == "appr":
        d = af.offset(ag, obs, j, tgt, second)
        if d is not None: ag.off = dict(tgt=tgt, d=d, tsw=TSW, on=True)
    elif j > 0:
        rng = np.random.default_rng([seed, leg, j]); ag.p_hat = ag.p_hat + rng.normal(0, 0.2, 3)
    t0 = float(obs["simulation_time"]); k0 = af.popped(obs); ag.hist = []
    while True:
        st0 = obs["balloon_status"][:, 0].copy(); ag.last = None
        done = step()
        if af.popped(obs) > k0:
            nb = [int(b) for b in np.flatnonzero((obs["balloon_status"][:, 0] == 2) & (st0 != 2))]
            events.append(dict(leg=leg, opt=j, t=float(obs["simulation_time"]), popped=nb, tgt=tgt, cur_target=ag.target,
                               off=None if ag.off is None else bool(ag.off["on"]), last=ag.last,
                               d=None if ag.off is None else float(np.linalg.norm(ag.off["d"])), hist_tail=ag.hist[-3:], n_hist=len(ag.hist),
                               last_on=[h for h in ag.hist if h[2]][-1:] ))
            break
        if done or not ag._burning(float(obs["simulation_time"])) or float(obs["simulation_time"]) - t0 > af.LEG_MAX:
            events.append(dict(leg=leg, opt=j, t=float(obs["simulation_time"]), popped=[], tgt=tgt, off=None if ag.off is None else bool(ag.off["on"])))
            break
    ag.off = None; leg += 1
ag.follow = False
while not done:
    k0 = af.popped(obs); st0 = obs["balloon_status"][:, 0].copy(); done = step()
    if af.popped(obs) > k0:
        events.append(dict(leg="post", t=float(obs["simulation_time"]), popped=[int(b) for b in np.flatnonzero((obs["balloon_status"][:, 0] == 2) & (st0 != 2))]))
json.dump(TR, open(f"./dev/gnc2_seq/verify_approach/tr_{seed}.json","w"))
print(json.dumps(dict(seed=seed, mode=mode, spec=spec, pops=af.popped(obs), events=events), default=str))

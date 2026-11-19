"""D2: the real worker_loop with a fake env / connection. The live policy gains the res head at env step 30 (picked up by
the reload before segment 2, so episode 1 = steps 1-100 spans widths 5 and 8), and act() is forced to return one 5-column
row inside segment 4 (a mixed-width segment). Every episode returns 100 (>= SIL_MIN) -> each is a demo candidate."""
import os, shutil, sys, time
from pathlib import Path

os.environ.update(RL3_ATTC="1", RL3_RES="0.25", SIL_MIN="10")
R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R))
import numpy as np  # noqa: E402
import dev.rl3_env  # noqa: E402
from dev import async_ppo as ap  # noqa: E402
from dev.res_check.fakeseg import obs  # noqa: E402

T = R / "dev/res_check/tmp"
run = T / "runD2"; shutil.rmtree(run, ignore_errors=True); run.mkdir()
base = dict(np.load(T / "rl3_policy_86630400.npz")); res = dict(np.load(T / "rl3_policy_resD.npz"))
np.savez(run / "live_policy.npz", **base, version=0); (run / "learner.addr").write_text("localhost 1\n")
X0 = obs(np.random.default_rng(0), 1)[0]
n_step, n_act, sent = [0], [0], []


class Stop(Exception):
    pass


class FakeEnv:
    def __init__(self, **kw):
        self.agent_kw, self.t = {}, 0

    def reset(self, seed=None):
        self.t = 0; return X0, {}

    def step(self, a):
        self.t += 1; n_step[0] += 1
        if n_step[0] == 30:
            time.sleep(0.01); np.savez(run / "live_policy.tmp.npz", **res, version=1); os.replace(run / "live_policy.tmp.npz", run / "live_policy.npz")
        return X0, 1.0, self.t >= 100, False, {"seed": 1, "last_pop": 1.0}


class FakeConn:
    def send(self, m):
        sent.append((n_step[0], m))
        if sum(x[1][0] == "segment" for x in sent) >= 6:
            raise Stop


act0 = ap.act
def act(agent, x, launch, rng):  # noqa: E302
    n_act[0] += 1; a, lp, v = act0(agent, x, launch, rng)
    return (a[:5] if 3 * 65 < n_act[0] <= 3 * 65 + 1 else a), lp, v  # one 5-column row in segment 4


dev.rl3_env.GuidanceEnv3 = FakeEnv; ap.Client = lambda *a, **k: FakeConn(); ap.act = act
try:
    ap.worker_loop(str(run), True, 0)
except Stop:
    print("worker ran without raising until 6 segments were sent")
segs = [(s, m[1], m[2]["act"].shape) for s, m in sent if m[0] == "segment"]
print("segments sent (env step at send, version, action shape):", segs)
print("demos sent at env step:", [s for s, m in sent if m[0] == "demo"], "widths", [m[1]["act"].shape for s, m in sent if m[0] == "demo"])
print("episodes ended at env step:", [s for s, m in sent if m[0] == "episode"])

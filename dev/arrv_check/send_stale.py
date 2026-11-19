"""E helper: once the learner of RUN is up, save its first live_policy.npz (the loaded weights, before any update) as
RUN_v0.npz and send it one stale 10-column segment and one 10-column demo (the action width of a 5-output res head + attc)
over its Listener, at the current version (so only the width check can drop them), and a 13-column segment / demo without the
"nr" stamp (a worker on pre-RES3 code flying an 8-output policy: same width, wrong meaning; async_ppo.fits drops them).  python send_stale.py RUN"""
import shutil, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np  # noqa: E402
from multiprocessing.connection import Client  # noqa: E402
from dev import async_ppo as ap  # noqa: E402
from dev.remote_envs import authkey  # noqa: E402
from dev.res_check.fakeseg import obs  # noqa: E402

run = Path(sys.argv[1])
while not (run / "learner.addr").exists() or not (run / "live_policy.npz").exists():
    time.sleep(1)
time.sleep(2)
shutil.copy(run / "live_policy.npz", str(run) + "_v0.npz")
ver = int(np.load(run / "live_policy.npz")["version"])
rng = np.random.default_rng(0)
X = obs(rng, ap.SEG); A = np.zeros((ap.SEG, 10)); A[:, 0] = rng.integers(0, 10, ap.SEG); A[:, 1] = rng.integers(0, 4, ap.SEG)
seg = {"obs": X, "act": A, "logp": np.zeros(ap.SEG), "val": np.zeros(ap.SEG), "rew": np.ones(ap.SEG), "done": np.zeros(ap.SEG, bool)}
host, port = (run / "learner.addr").read_text().split()
c = Client((host, int(port)), authkey=authkey())
c.send(("segment", ver, seg, 0.0))
c.send(("demo", {"obs": X, "act": A, "rew": np.ones(ap.SEG, np.float32)}))
A13 = np.concatenate([A, np.zeros((ap.SEG, 3))], 1)
c.send(("segment", ver, {**seg, "act": A13}, 0.0))
c.send(("demo", {"obs": X, "act": A13, "rew": np.ones(ap.SEG, np.float32)}))
print(f"sent stale segments and demos of action shape {A.shape} and {A13.shape} (unstamped) at version {ver}", flush=True)
time.sleep(5)

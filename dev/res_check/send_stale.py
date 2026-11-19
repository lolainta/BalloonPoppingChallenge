"""Check 6 helper: once the learner of RUN is up, save its first live_policy.npz (the loaded weights, before any update) and
send it one stale 8-column segment (a 3-output-head width) over its Listener."""
import os, shutil, sys, time
from pathlib import Path

os.environ.update(RL3_ATTC="1", RL3_RES="0.25", RL3_RES2="1")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np  # noqa: E402
from multiprocessing.connection import Client  # noqa: E402
from dev.remote_envs import authkey  # noqa: E402

run = Path(sys.argv[1])
while not (run / "learner.addr").exists():
    time.sleep(1)
time.sleep(2)
shutil.copy(run / "live_policy.npz", str(run) + "_v0.npz")
d = dict(np.load(run / "live_policy.npz")); ver = int(d["version"])
os.environ["RL3_RES2"] = ""  # build the stale segment with a 3-output head policy (8 columns)
import dev.async_ppo  # noqa: E402,F401
from dev.res_check.fakeseg import segment  # noqa: E402
host, port = (run / "learner.addr").read_text().split()
c = Client((host, int(port)), authkey=authkey())
old = str(run) + "_old3.npz"; np.savez(old, **{k: (v[:, :3] if k == "res.w" else v[:3] if k in ("res.b", "res.logstd") else v) for k, v in d.items()})
seg = segment(old, np.random.default_rng(0), 8)
c.send(("segment", ver, seg, 0.0))
print(f"sent a stale segment of action shape {seg['act'].shape} at version {ver}", flush=True)
time.sleep(5)

"""D1: a real learner warm-started with RL3_RES (no local workers), fed over its Listener by a fake worker: one stale
5-column demo, one stale 5-column segment, then 7 normal 8-column segments -> must drop the stale ones and update."""
import os, shutil, subprocess, sys, time
from pathlib import Path

ENV = dict(RL3_SCEN="4", RL3_ATTC="1", RL3_T_MIN="1", RL3_FREE_LAUNCH="1", RL3_KEEP_LAUNCH="1", RL3_RES="0.25",
           GNC_KW="k_att:3+max_rate:2.5+coast:1+min_alt:3+lag_comp:1+bwind:4+margin:0+eob:2+eob_tau:3+eob_T:4")
os.environ.update(ENV)
R = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(R))
import numpy as np  # noqa: E402
from multiprocessing.connection import Client  # noqa: E402
from dev.remote_envs import authkey  # noqa: E402
from dev.res_check.fakeseg import segment  # noqa: E402

run = R / "dev/res_check/tmp/runD1"
shutil.rmtree(run, ignore_errors=True); run.mkdir()
shutil.copy(R / "dev/res_check/tmp/base_async_last.pt", run / "async_last.pt")
log = open(str(run) + ".log", "w")
L = subprocess.Popen([str(R / ".venv-gpu/bin/python"), str(R / "dev/async_ppo.py"), "learn", "--run", str(run), "--launch", "--local", "0",
                      "--batch", "400", "--ckpt_every", "0"], stdout=log, stderr=subprocess.STDOUT, cwd=R, env={**os.environ, "RES_CHECK_RUN": str(run)})
try:
    while not (run / "learner.addr").exists():
        time.sleep(1)
    time.sleep(1)
    host, port = (run / "learner.addr").read_text().split()
    c = Client((host, int(port)), authkey=authkey())
    live = run / "live_policy.npz"; ver = int(np.load(live)["version"])
    rng = np.random.default_rng(0)
    old = segment(live, rng, 5)
    c.send(("demo", {"obs": old["obs"], "act": old["act"], "rew": old["rew"].astype(np.float32)}))
    c.send(("segment", ver, old, 0.0))
    for _ in range(7):
        c.send(("segment", ver, segment(live, rng, 8), 0.0))
    t0 = time.time()
    while time.time() - t0 < 120 and "steps" not in Path(str(run) + ".log").read_text():
        time.sleep(1)
    time.sleep(2)
    print("learner alive after the update:", L.poll() is None)
finally:
    L.terminate(); L.wait(10)
print("".join(l for l in open(str(run) + ".log") if not any(w in l for w in ("Warn", "detach", "newent {"))))

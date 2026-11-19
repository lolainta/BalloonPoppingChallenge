"""A / D: a deterministic CPU learner (async_ppo.learn of SRC, torch / numpy RNGs seeded) fed by an in-process sender with
fixed batches: the 64-row segments of ROWS (worker rows of the s4res2 policy), BATCH / 64 per update in a fixed cycle,
tagged with the learner's current version, plus every finished episode of ROWS as a SIL demo once. After NUPD updates
(counted from the start / resume) the checkpoint is copied to OUT and the process exits. No workers (--local 0).
source dev/hoent_check/env.sh; $GPY dev/hoent_check/det_learn.py SRC.py RUN NUPD OUT.pt ROWS.npz[,ROWS2.npz..] [learner args]"""
import os
import runpy, shutil, sys, threading, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
src, run, nupd, out, rows = sys.argv[1:6]
run, nupd = Path(run), int(nupd)
R = [dict(np.load(f)) for f in rows.split(",")]
segs, demos = [], []
for r in R:
    n = len(r["obs"]) // 64 * 64
    segs += [{k: r[k][i: i + 64] for k in ("obs", "act", "logp", "val", "rew", "done")} for i in range(0, n, 64)]
    e = np.flatnonzero(r["done"]) + 1
    demos += [{"obs": r["obs"][a:b].astype(np.float32), "act": r["act"][a:b], "rew": r["rew"][a:b].astype(np.float32)}
              for a, b in zip(np.r_[0, e[:-1]], e)]


def sender():
    from multiprocessing.connection import Client
    from dev.remote_envs import authkey
    while not (run / "learner.addr").exists() or not (run / "live_policy.npz").exists():
        time.sleep(0.2)
    host, port = (run / "learner.addr").read_text().split()
    c = Client((host, int(port)), authkey=authkey())
    v0 = int(np.load(run / "live_policy.npz")["version"]); k = 0
    for d in demos:
        c.send(("demo", d))
    for u in range(nupd):
        while int(np.load(run / "live_policy.npz")["version"]) < v0 + u:
            time.sleep(0.05)
        for _ in range(B // 64):
            c.send(("segment", v0 + u, segs[k % len(segs)], 0.0)); k += 1
    ck = run / "async_last.pt"
    while True:
        try:
            if torch.load(ck, map_location="cpu")["version"] == v0 + nupd:
                break
        except Exception:
            pass
        time.sleep(0.2)
    time.sleep(1); shutil.copy(ck, out)
    print(f"[det_learn] {nupd} updates from version {v0}: checkpoint -> {out}", flush=True)
    os._exit(0)


argv = sys.argv[6:]
B = int(argv[argv.index("--batch") + 1]) if "--batch" in argv else 256
for f in ("learner.addr", "live_policy.npz"):  # a restart: the previous learner's address / version are stale
    (run / f).unlink(missing_ok=True)
torch.manual_seed(0); np.random.seed(0); torch.use_deterministic_algorithms(True)
threading.Thread(target=sender, daemon=True).start()
sys.argv = [src, "learn", "--run", str(run), "--launch", "--local", "0", "--batch", str(B), "--ckpt_every", "0"] + \
    [x for i, x in enumerate(argv) if x != "--batch" and (i == 0 or argv[i - 1] != "--batch")]
runpy.run_path(src, run_name="__main__")

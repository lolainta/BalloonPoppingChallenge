"""Hybrid E2E M3 (DECISION_LOG §262): recurrent PPO for a GRU thrust correction on the frozen s40a agent (dev/e2e.py fly res=True;
the RL3 policy still launches, picks the targets and flies the GNC, the net adds RES_B * a_thrust * tanh(d) every K steps).
  learner: python dev/e2e_ppo.py learn RUN       (GPU if available; listens, writes RUN/learner.addr)
  workers: python dev/e2e_ppo.py work RUN N      (N episode processes, each: latest weights -> one episode -> send it back)
The learner updates once B episodes at most MAX_LAG versions old are in (older ones dropped); the first WARM updates train only
the critic (the policy stays exactly s40a: mean correction 0). Every episode's seed goes to RUN/episode_seeds.csv (scenario 40)."""
import csv
import json
import os
import queue
import socket
import sys
import threading
import time
from multiprocessing.connection import Client, Listener
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dev import e2e

KEY = (Path.home() / ".bpc_remote_key").read_bytes()  # (= dev/remote_envs.authkey, without its stable_baselines3 import)
B = int(os.environ.get("E2E_B", 32))  # episodes per update
MB = int(os.environ.get("E2E_MB", 8))  # episodes per minibatch
EPOCHS = int(os.environ.get("E2E_EPOCHS", 4))
MAX_LAG = int(os.environ.get("E2E_MAX_LAG", 3))
WARM = int(os.environ.get("E2E_WARM", 10))
LR = float(os.environ.get("E2E_LR", 3e-4))
SD0 = float(os.environ.get("E2E_SD0", 0.1))  # initial exploration std of d (correction ~ RES_B * SD0 * a_thrust)
CLIP, LAM, ENT = 0.2, 0.95, 0.0


def gae(R, V, gamma=e2e.GAMMA, lam=LAM):  # one whole episode (terminal value 0) -> advantages, returns
    A, g = np.zeros_like(R), 0.0
    for t in range(len(R) - 1, -1, -1):
        g = R[t] + gamma * (V[t + 1] if t + 1 < len(R) else 0.0) - V[t] + gamma * lam * g; A[t] = g
    return A, A + V


def learn(run):
    import torch
    run = Path(run); run.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = e2e.make_net().to(dev)
    with torch.no_grad():
        net.mu.weight.zero_(); net.mu.bias.zero_(); net.logstd.fill_(float(np.log(SD0)))
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    ver, n_eps, dropped = 0, 0, 0
    if (run / "last.pt").exists():
        ck = torch.load(run / "last.pt", map_location=dev)
        net.load_state_dict(ck["net"]); opt.load_state_dict(ck["opt"]); ver, n_eps = ck["ver"], ck["eps"]
    pack = lambda: (ver, {k: v.cpu().numpy() for k, v in net.state_dict().items()})  # noqa: E731
    shared, inbox = {"w": pack()}, queue.Queue()
    lis = Listener(("0.0.0.0", 0), authkey=KEY)
    (run / "learner.addr").write_text(f"{socket.gethostname()}:{lis.address[1]}")

    def serve(conn):  # worker -> ("hi" | "ep", episode or None, its version); reply: the latest weights if newer, else None
        try:
            while True:
                kind, ep, wv = conn.recv()
                if kind == "ep":
                    inbox.put(ep)
                w = shared["w"]
                conn.send(w if w[0] != wv else None)
        except (EOFError, OSError):
            pass

    def accept():
        while True:
            threading.Thread(target=serve, args=(lis.accept(),), daemon=True).start()

    threading.Thread(target=accept, daemon=True).start()
    seedw = csv.writer(sf := open(run / "episode_seeds.csv", "a", newline=""))
    if sf.tell() == 0:
        seedw.writerow(["seed", "scenario", "hour", "version", "pops", "time"])
    prog = open(run / "progress.csv", "a")
    if prog.tell() == 0:
        prog.write("version,eps,mean_pops,dropped,vloss,kl,clipfrac,std,time\n")
    buf, pops_all = [], []
    print(f"learner {dev} v{ver} on {socket.gethostname()}:{lis.address[1]}", flush=True)
    while True:
        ep = inbox.get(); n_eps += 1
        seedw.writerow([ep["seed"], 40, ep["hour"], ep["version"], ep["pops"], int(time.time())]); sf.flush()
        if ep["pops"] < 0 or len(ep["X"]) < 2:  # (an environment failure: the seed is recorded, nothing to learn)
            continue
        if ver - ep["version"] > MAX_LAG:
            dropped += 1; continue
        buf.append(ep); pops_all.append(ep["pops"])
        if len(buf) < B:
            continue
        st = update(net, opt, buf, dev, policy=ver >= WARM)
        ver += 1; shared["w"] = pack()
        prog.write(f"{ver},{n_eps},{np.mean(pops_all):.3f},{dropped},{st['vl']:.4f},{st['kl']:.5f},{st['cf']:.3f},"
                   f"{float(net.logstd.detach().exp().mean()):.4f},{int(time.time())}\n"); prog.flush()
        (run / "status.json").write_text(json.dumps(dict(version=ver, eps=n_eps, dropped=dropped, updated=int(time.time()))))
        buf, pops_all = [], []
        if ver % 10 == 0:
            ck = dict(net=net.state_dict(), opt=opt.state_dict(), ver=ver, eps=n_eps)
            torch.save(ck, run / ".last.pt"); (run / ".last.pt").replace(run / "last.pt")
            if ver % 50 == 0:
                torch.save({k: v.cpu() for k, v in net.state_dict().items()}, run / f"net_{ver}.pt")


def update(net, opt, buf, dev, policy=True):
    import torch
    T = max(len(e["X"]) for e in buf); n = len(buf)
    X = np.zeros((n, T, e2e.D), np.float32); A = np.zeros((n, T, 3), np.float32)
    LP, AD, RET, Mk = (np.zeros((n, T), np.float32) for _ in range(4))
    for i, e in enumerate(buf):
        m = len(e["X"]); ad, ret = gae(e["R"], e["V"])
        X[i, :m], A[i, :m], LP[i, :m], AD[i, :m], RET[i, :m], Mk[i, :m] = e["X"], e["U"], e["LP"], ad, ret, 1
    AD = (AD - AD[Mk > 0].mean()) / (AD[Mk > 0].std() + 1e-8) * Mk
    X, A, LP, AD, RET, Mk = (torch.from_numpy(a).to(dev) for a in (X, A, LP, AD, RET, Mk))
    st = dict(vl=0.0, kl=0.0, cf=0.0); k = 0
    for _ in range(EPOCHS):
        for idx in torch.randperm(n).split(MB):
            _, mu, v, _ = net(X[idx]); ls = net.logstd
            lp = (-0.5 * ((A[idx] - mu) / ls.exp()) ** 2 - ls - 0.5 * np.log(2 * np.pi)).sum(-1)
            r = (lp - LP[idx]).exp(); m = Mk[idx]; ms = m.sum()
            pl = -(torch.min(r * AD[idx], r.clamp(1 - CLIP, 1 + CLIP) * AD[idx]) * m).sum() / ms
            vl = (((v - RET[idx]) ** 2) * m).sum() / ms
            loss = (pl - ENT * ls.sum() if policy else 0.0) + 0.5 * vl
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5); opt.step()
            with torch.no_grad():
                st["vl"] += float(vl); st["kl"] += float(((LP[idx] - lp) * m).sum() / ms)
                st["cf"] += float((((r - 1).abs() > CLIP).float() * m).sum() / ms); k += 1
    return {a: b / k for a, b in st.items()}


def worker(run, i):
    import torch
    from dev.run_seeds import parse_kw
    torch.set_num_threads(1)
    import contextlib, io, logging, warnings  # noqa: E401
    warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
    net, ver, rng = e2e.make_net(), -1, np.random.default_rng()
    kw = parse_kw(os.environ.get("E2E_KW", "").split())
    msg, wait = ("hi", None, ver), 1.0
    while True:
        try:
            host, port = (Path(run) / "learner.addr").read_text().split(":")
            conn = Client((host, int(port)), authkey=KEY)
            while True:
                conn.send(msg)
                w = conn.recv(); wait = 1.0
                if w is not None:
                    ver = w[0]; net.load_state_dict({k: torch.from_numpy(v) for k, v in w[1].items()})
                seed, hour = int(rng.integers(2**31)), int(rng.choice(e2e.HOURS))
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        ep = e2e.fly(seed, hour, net=net, res=True, sample=True, kw=dict(kw), rng=rng)
                    ep = {k: ep[k] for k in ("X", "U", "LP", "V", "R", "pops")}
                except Exception as ex:  # noqa: BLE001 (a bad environment draw: the seed is still recorded)
                    print(f"worker {i}: seed {seed}: {ex!r}", flush=True); ep = dict(X=np.zeros((0, e2e.D)), pops=-1)
                msg = ("ep", dict(ep, seed=seed, hour=hour, version=ver), ver)
        except (EOFError, OSError, ValueError):
            time.sleep(wait); wait = min(wait * 2, 60.0)  # (the learner restarts: same message again)


if __name__ == "__main__":
    if sys.argv[1] == "learn":
        learn(sys.argv[2])
    elif sys.argv[1] == "work":
        import multiprocessing as mp
        ps = [mp.get_context("spawn").Process(target=worker, args=(sys.argv[2], i)) for i in range(int(sys.argv[3]))]
        [p.start() for p in ps]; [p.join() for p in ps]

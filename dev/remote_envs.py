"""Multi-node PPO rollouts: env workers on extra Slurm nodes + one VecEnv that merges them with the local envs.

Worker (one per extra node):  .venv/bin/python dev/remote_envs.py serve --run runs/ppo5 --n 15 [--launch]
  starts a local SubprocVecEnv, listens on a free port and writes <run>/workers/<host>_<port>.addr
Learner: train_ppo2.py --remote <run>/workers  merges every live worker listed there into its VecEnv.

Synchronous like SubprocVecEnv: each step waits for the slowest env (local or remote). Auth: a random key in
~/.bpc_remote_key (0600, shared home), so only our own processes can connect.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse, glob, secrets, socket, sys, time
from multiprocessing.connection import Client, Listener
from pathlib import Path

import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
KEY_FILE = Path.home() / ".bpc_remote_key"


def authkey():
    if not KEY_FILE.exists():
        KEY_FILE.write_bytes(secrets.token_bytes(32))
        KEY_FILE.chmod(0o600)
    return KEY_FILE.read_bytes()


def make_env_fn(launch):
    if launch:
        from dev.rl3_env import GuidanceEnv3 as E
    else:
        from dev.rl2_env import GuidanceEnv as E
    return lambda: E()


def serve(run, n, launch):
    venv = SubprocVecEnv([make_env_fn(launch) for _ in range(n)], start_method="fork")
    lis = Listener(("0.0.0.0", 0), authkey=authkey())
    host, port = socket.gethostname().split(".")[0], lis.address[1]
    d = Path(run) / "workers"; d.mkdir(parents=True, exist_ok=True)
    addr = d / f"{host}_{port}.addr"
    addr.write_text(f"{host} {port} {n} {os.environ.get('SLURM_JOB_ID', '')}\n")
    print(f"serving {n} envs on {host}:{port}", flush=True)
    try:
        while True:
            conn = lis.accept()  # one learner at a time; a new learner (e.g. after resume) reconnects
            print("learner connected", flush=True)
            try:
                while True:
                    cmd, *args = conn.recv()
                    if cmd == "spaces":
                        conn.send((venv.observation_space, venv.action_space, venv.num_envs))
                    elif cmd == "seed":
                        venv.seed(args[0]); conn.send(None)
                    elif cmd == "reset":
                        conn.send((venv.reset(), venv.reset_infos))
                    elif cmd == "step":
                        conn.send(venv.step(args[0]))
                    elif cmd == "get_attr":
                        conn.send(venv.get_attr(args[0]))
                    elif cmd == "close":
                        break
            except (EOFError, ConnectionError):
                pass
            print("learner disconnected", flush=True)
    finally:
        addr.unlink(missing_ok=True)
        venv.close()


class MultiNodeVecEnv(VecEnv):
    """Local SubprocVecEnv + remote workers, presented as one VecEnv (local envs first)."""

    def __init__(self, local, worker_dir):
        self.local, self.remotes, self.sizes = local, [], [local.num_envs]
        for f in sorted(glob.glob(f"{worker_dir}/*.addr")):
            host, port, n, *_ = Path(f).read_text().split()
            try:
                c = Client((host, int(port)), authkey=authkey())
                c.send(("spaces",)); _, _, n_remote = c.recv()
            except OSError as e:
                print(f"skip dead worker {f}: {e}", flush=True)
                continue
            self.remotes.append(c); self.sizes.append(n_remote)
            print(f"connected {host}:{port} ({n_remote} envs)", flush=True)
        self.bounds = np.cumsum([0] + self.sizes)
        super().__init__(int(self.bounds[-1]), local.observation_space, local.action_space)

    def _split(self, arr):
        return [arr[self.bounds[i]: self.bounds[i + 1]] for i in range(len(self.sizes))]

    def reset(self):
        seeds = self._seeds if any(s is not None for s in self._seeds) else None
        if seeds is not None:
            for part, venv in zip(self._split(np.array(seeds, dtype=object)), [self.local] + self.remotes):
                s0 = part[0]
                if venv is self.local:
                    venv.seed(s0)
                else:
                    venv.send(("seed", s0)); venv.recv()
        for c in self.remotes:
            c.send(("reset",))
        obs = [self.local.reset()]
        infos = list(self.local.reset_infos)
        for c in self.remotes:
            o, i = c.recv(); obs.append(o); infos += list(i)
        self.reset_infos = infos
        self._reset_seeds(); self._reset_options()
        return np.concatenate(obs)

    def step_async(self, actions):
        parts = self._split(actions)
        for c, a in zip(self.remotes, parts[1:]):
            c.send(("step", a))
        self.local.step_async(parts[0])

    def step_wait(self):
        res = [self.local.step_wait()] + [c.recv() for c in self.remotes]
        obs, rew, done, info = zip(*res)
        return np.concatenate(obs), np.concatenate(rew), np.concatenate(done), sum((list(i) for i in info), [])

    def close(self):
        for c in self.remotes:
            try:
                c.send(("close",)); c.close()
            except OSError:
                pass
        self.local.close()

    def get_attr(self, name, indices=None):
        vals = list(self.local.get_attr(name))
        for c in self.remotes:
            c.send(("get_attr", name)); vals += list(c.recv())
        return [vals[i] for i in self._get_indices(indices)]

    def set_attr(self, name, value, indices=None):
        raise NotImplementedError

    def env_method(self, name, *args, indices=None, **kw):
        raise NotImplementedError

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * len(list(self._get_indices(indices)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["serve"])
    ap.add_argument("--run", required=True)
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--launch", action="store_true")
    a = ap.parse_args()
    serve(a.run, a.n, a.launch)

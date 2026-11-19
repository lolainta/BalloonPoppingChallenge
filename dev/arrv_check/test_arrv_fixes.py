"""Fast tests of the RL3_RES3 verifier fixes (no env).  PYTHONPATH=<tree> python dev/arrv_check/test_arrv_fixes.py
(plain asserts, pytest-compatible; the venvs have no pytest)
1) _arrv: held / changed dv -> arrival velocity + ~(the last) dv (N = 2: 0.98 dv from 3 s out), miss unchanged (open loop, point mass); 0 in the coast branch.
2) exploration: the dv dims of a new / grown res head start at std RL3_RES_VSTD (0.1), the others at 0.3.
3) learner guard: an 8-output run takes only segments stamped nr = 8 (a worker on pre-RES3 code sends 13-wide rows unstamped)."""
import contextlib
import importlib
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from BalloonPoppingGymEnv.agents import gnc_agent  # noqa: E402
from BalloonPoppingGymEnv.agents.gnc_agent import G  # noqa: E402

assert Path(gnc_agent.__file__).resolve().is_relative_to(ROOT)


class Fake(gnc_agent.GNCAgent):  # just the attributes _arrv reads
    def __init__(self, dt, N=2.0):
        self.dv, self.vcap, self.a_thrust, self.stretch, self.s_mul, self.t_go_min = np.zeros(3), 0.5, 14.7, 1.0, 1.0, 0.3
        self.dt, self.nav_N, self.n_mul, self.coast, self.burn_left, self.target = dt, N, 1.0, 0, 9.0, 0
        self.dv_p, self.dv_v, self.dv_tgt = np.zeros(3), np.zeros(3), None


def fly(dvs, T=3.0, dt=0.002, N=2.0):
    """dvs: [(time to go, dv)]: dv set at those times (decisions). -> miss, arrival velocity."""
    p, v = np.zeros(3), np.array([2.0, 0.0, 12.0])
    b0, bv = np.array([15.0, -8.0, 30.0]), np.array([0.5, 0.3, 1.5])
    f = Fake(dt, N)
    for k in range(int(round(T / dt))):
        tg = T - k * dt
        for tc, d in dvs:
            if abs(tg - tc) < dt / 2:
                f.dv = np.asarray(d, float)
        zem = b0 + bv * T - (p + v * tg + 0.5 * G * tg**2)
        a = N * zem / tg**2
        if f.dv.any() or f.dv_v.any() or f.dv_p.any():
            a = a + f._arrv(tg)
        acc = a + G
        p = p + v * dt + 0.5 * acc * dt**2; v = v + acc * dt
    return np.linalg.norm(p - (b0 + bv * T)), v


def test_held_dv_moves_arrival_by_dv():
  for N in (1.7, 2.0, 2.3):  # nav_N 2 * n_mul (exp(+-0.15))
    _, v0 = fly([], N=N)
    for d in ([1.0, -1.0, 1.0], [0.0, 0.0, -3.0]):
        dvs = [(3.0 - 0.5 * i, d) for i in range(6)]  # decided again every 0.5 s with the same dv: must not add up
        miss, v1 = fly(dvs, N=N)
        assert miss < 1e-3
        assert np.allclose(v1 - v0, d, atol=(0.03 if N == 2.0 else 0.15) * np.linalg.norm(d)), (N, v1 - v0, d)  # (after the fade: N's)


def test_changed_dv_ends_at_the_last_one():
    _, v0 = fly([])
    miss, v1 = fly([(3.0, [1.0, -1.0, 1.0]), (2.5, [0.0, 2.0, 0.0]), (2.0, [0.0, 0.0, 0.0])])
    assert miss < 1e-3 and np.abs(v1 - v0).max() < 0.05


def test_coast_branch_zero():
    f = Fake(0.01); f.dv = np.array([1.0, 2.0, 3.0]); f.coast, f.burn_left = 1, 1.0
    assert not f._arrv(2.0).any() and not f.dv_v.any()
    assert f._arrv(0.9).any()


def test_new_target_resets_deviation():
    f = Fake(0.01); f.dv = np.array([1.0, 0.0, 0.0])
    for _ in range(50):
        f._arrv(2.0)
    assert f.dv_v.any()
    f.target = 1; d0 = f._arrv(2.0)
    assert np.allclose(d0, -2.0 * f.dv / 2.0)  # first step on a target: -2 dv / t (dv_p = dv_v = 0)


@contextlib.contextmanager
def env(**kw):  # RL3_* env vars for a test (None: unset), restored after
    old = {k: os.environ.get(k) for k in kw}
    try:
        for k, v in kw.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
        yield
    finally:
        for k, v in old.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


RES3 = dict(RL3_RES="0.25", RL3_RES2="1", RL3_RES3="1", RL3_RES_VSTD=None, RL3_TCOND=None)


def test_dv_std_init():
  import torch
  from dev import rl3_net as res3
  assert Path(res3.__file__).resolve().is_relative_to(ROOT)
  with env(**RES3), tempfile.TemporaryDirectory() as tmp_path:
    tmp_path = Path(tmp_path)
    net = res3.Net3()
    ls = net.res_logstd.detach().numpy()
    assert np.allclose(ls[:5], np.log(0.3)) and np.allclose(ls[5:], np.log(0.1))
    res3.export(net, tmp_path / "a.npz")  # a 5-output npz grown by load_npz: new dims ln 0.1 too
    P = dict(np.load(tmp_path / "a.npz"))
    P["res.w"], P["res.b"], P["res.logstd"] = P["res.w"][:, :5], P["res.b"][:5], P["res.logstd"][:5]
    np.savez(tmp_path / "b.npz", **P)
    net2 = res3.Net3(); res3.load_npz(net2, tmp_path / "b.npz")
    with torch.no_grad():
        assert np.allclose(net2.res_logstd.numpy(), ls)


def test_learner_takes_only_stamped_8_output_rows():
  with env(**RES3):
    from dev import async_ppo as ap
    assert Path(ap.__file__).resolve().is_relative_to(ROOT)
    ap = importlib.reload(ap)  # NR / RES from this env
    assert ap.RES and ap.NR == 8
    m = {"act": np.zeros((4, 13))}
    assert not ap.fits(m, 13)  # pre-RES3 worker: same width, u[3:8] flown as res / s_mul / n_mul
    assert not ap.fits({**m, "nr": 5}, 13)
    assert ap.fits({**m, "nr": 8}, 13)
    assert not ap.fits({"act": np.zeros((4, 10)), "nr": 8}, 13)
    from dev.rl3_env import res_cols
    assert res_cols({"res_bound": 0.25, "res_sbound": 0.3, "res_vbound": 10.0}) == 8 and res_cols({"res_sbound": 0.3}) == 5


def test_older_runs_unchanged():
  with env(RL3_RES="0.25", RL3_RES2="1", RL3_RES3=None):
    from dev import async_ppo as ap
    ap = importlib.reload(ap)
    assert ap.NR == 5 and ap.fits({"act": np.zeros((4, 10))}, 10)  # no stamp needed below 8 outputs


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name, flush=True)

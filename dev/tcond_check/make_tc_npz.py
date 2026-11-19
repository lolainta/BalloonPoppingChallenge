"""B/C/E: a res-head checkpoint -> the same weights + a tc head via rl3_net.load_npz / export: zero (as a learner adds it),
or random N(0, SCALE^2) weights / biases (seed 0) for the non-zero checks.
RL3_TCOND=1 RL3_RES=0.25 RL3_RES2=1 RL3_ATTC=1 RL3_T_MIN=1 GNC_KW=.. python dev/tcond_check/make_tc_npz.py BASE.npz OUT.npz [SCALE]"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dev import rl3_net  # noqa: E402
from dev.rl3_net import Net3, export, load_npz  # noqa: E402

assert Path(rl3_net.__file__).resolve().is_relative_to(ROOT), rl3_net.__file__
base, out = sys.argv[1:3]
scale = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
net = Net3(); load_npz(net, base); export(net, out)
B, O = dict(np.load(base)), dict(np.load(out))
same = all(k in O and np.array_equal(B[k], O[k]) for k in B)
if scale:
    r = np.random.default_rng(0)
    O["tc.w"] = (scale * r.standard_normal(O["tc.w"].shape)).astype(np.float32)
    O["tc.b"] = (scale * r.standard_normal(O["tc.b"].shape)).astype(np.float32)
    np.savez(out, **O)
print("base keys identical:", same, " new keys:", {k: (O[k].shape, float(np.abs(O[k]).max())) for k in O if k not in B})

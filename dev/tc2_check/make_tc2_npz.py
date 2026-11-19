"""B/C/E: an 8-output res + tc checkpoint -> the same weights + a tc2 head via rl3_net.load_npz / export: zero (as a learner adds
it), or random N(0, SCALE^2) weights and N(0, BSCALE^2) biases (seed 0; default BSCALE = SCALE) for the non-zero checks.
source dev/tc2_check/env.sh;  $PY dev/tc2_check/make_tc2_npz.py BASE.npz OUT.npz [SCALE [BSCALE]]"""
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
bscale = float(sys.argv[4]) if len(sys.argv) > 4 else scale
net = Net3(); assert net.tc2 is not None, "RL3_TCOND2=1"
load_npz(net, base); export(net, out)
B, O = dict(np.load(base)), dict(np.load(out))
same = all(k in O and np.array_equal(B[k], O[k]) for k in B)
if scale or bscale:
    r = np.random.default_rng(0)
    O["tc2.w"] = (scale * r.standard_normal(O["tc2.w"].shape)).astype(np.float32)
    O["tc2.b"] = (bscale * r.standard_normal(O["tc2.b"].shape)).astype(np.float32)
    np.savez(out, **O)
print("base keys identical:", same, " new keys:", {k: (O[k].shape, float(np.abs(O[k]).max())) for k in O if k not in B})

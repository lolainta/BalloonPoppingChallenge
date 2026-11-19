"""B/C/D/F: a 5-output (+tc) checkpoint -> the same weights with an 8-output res head via rl3_net.load_npz / export (new rows
zero, as a learner grows it), or with random N(0, SCALE^2) new rows (res u[5:8] and their tc columns; seed 0) for the
non-zero checks (biases N(0, BSCALE^2), default BSCALE = 10 SCALE). RL3_TCOND follows the base. (source dev/arrv_check/env.sh)  python make_arrv_npz.py BASE.npz OUT.npz [SCALE [BSCALE]]"""
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
base, out = sys.argv[1:3]
scale = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
bscale = float(sys.argv[4]) if len(sys.argv) > 4 else 10.0 * scale
B = dict(np.load(base))
os.environ["RL3_RES3"] = "1"
os.environ.pop("RL3_TCOND", None)
if "tc.w" in B:
    os.environ["RL3_TCOND"] = "1"
from dev import rl3_net  # noqa: E402
from dev.rl3_net import Net3, export, load_npz  # noqa: E402

assert Path(rl3_net.__file__).resolve().is_relative_to(ROOT), rl3_net.__file__
net = Net3(); load_npz(net, base); export(net, out)
O = dict(np.load(out))
grown = {"res.w": 5, "res.b": 5, "res.logstd": 5, "tc.w": 9, "tc.b": 9}
same = all(k in O and np.array_equal(B[k], O[k][..., : grown[k]] if k in grown else O[k]) for k in B)
zero = all(not O[k][..., grown[k]:].any() for k in grown if k in O and k != "res.logstd")
if scale or bscale:
    r = np.random.default_rng(0)
    for k in ("res.w", "res.b", "tc.w", "tc.b"):
        if k in O:
            s = bscale if k.endswith(".b") else scale
            O[k][..., grown[k]:] = (s * r.standard_normal(O[k][..., grown[k]:].shape)).astype(np.float32)
    np.savez(out, **O)
print("old entries identical:", same, " new rows zero (before SCALE):", zero, " res.logstd:", O["res.logstd"].round(4),
      " shapes:", {k: O[k].shape for k in grown if k in O}, " new keys:", {k: float(O[k]) for k in O if k not in B})

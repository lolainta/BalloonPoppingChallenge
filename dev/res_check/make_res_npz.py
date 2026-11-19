"""B: base checkpoint -> the same weights + a zero-initialised res head, via rl3_net.load_npz / export.
RL3_RES=0.25 RL3_ATTC=1 RL3_T_MIN=.. GNC_KW=.. python dev/res_check/make_res_npz.py BASE.npz OUT.npz"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dev.rl3_net import Net3, export, load_npz  # noqa: E402

base, out = sys.argv[1:3]
net = Net3(); load_npz(net, base); export(net, out)
B, O = dict(np.load(base)), dict(np.load(out))
same = all(k in O and np.array_equal(B[k], O[k]) for k in B)
print("base keys identical:", same, " new keys:", {k: O[k].tolist() if O[k].size <= 3 else (O[k].shape, float(np.abs(O[k]).max())) for k in O if k not in B})

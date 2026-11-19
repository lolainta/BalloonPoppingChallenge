"""Defect-1 diagnosis: which weights carry a target-entropy change between INIT.npz and AFTER.npz: the entropy (ent_probe rows) of AFTER
with one group of keys put back to INIT's values, and the relative change of each group.
source dev/fast_check/env.sh; $PY dev/fast_check/ent_swap.py PROBE.npy INIT.npz AFTER.npz..."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from BalloonPoppingGymEnv.agents import rl3_agent  # noqa: E402

assert Path(rl3_agent.__file__).resolve().is_relative_to(ROOT)
X = np.load(sys.argv[1])[:256]
I = dict(np.load(sys.argv[2]))
GROUPS = {"target head (tgt, prior)": ("tgt.", "prior"), "token encoder + global (t0, t1, g)": ("t0.", "t1.", "g."),
          "attention blocks": ("l0.", "l1.", "l2.", "l3."), "value (v0, v1)": ("v0.", "v1.")}


def ent(P):
    ag = rl3_agent.RL3Agent.__new__(rl3_agent.RL3Agent); ag.net = P; H = []
    for x in X:
        tl = ag.forward(x)[0]; z = tl - tl.max(); p = np.exp(z) / np.exp(z).sum()
        H.append(-(p * np.log(np.maximum(p, 1e-30))).sum())
    return float(np.mean(H))


print(f"init {ent(I):.4f}")
for path in sys.argv[3:]:
    A = dict(np.load(path))
    row = [f"{Path(path).parent.name}/{Path(path).name}: {ent(A):.4f}"]
    for g, pre in GROUPS.items():
        keys = [k for k in I if k.startswith(pre) and k in A and A[k].shape == I[k].shape]
        rel = np.sqrt(sum(((A[k] - I[k]) ** 2).sum() for k in keys) / max(sum((I[k] ** 2).sum() for k in keys), 1e-30))
        row.append(f"[{g}: back to init -> {ent({**A, **{k: I[k] for k in keys}}):.4f}, rel change {rel:.1e}]")
    print("  ".join(row), flush=True)

"""Defect-3 check: RL3_ATT_RAND only overwrites the attitude columns of a policy with an attitude head (async_ppo.has_att, the
worker's condition); a fast + res policy without one keeps its pad action [0, 3, j, flag 1, u (8)] (before: a[3] = the flag column).
$PY dev/fast_check/check_att.py  (RL3_ATTC as the caller sets it)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dev import async_ppo as ap  # noqa: E402

assert Path(ap.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[2])
for keys in ({"res.w", "fast.w"}, {"res.w", "fast.w", "attc.w"}, {"res.w", "att.w"}, {"att.w"}, {"attc.w"}, set()):
    a = [0, 3, 1] + ([0.0, 0.0] if "attc.w" in keys else [0] if "att.w" in keys else []) + ([1.0] if "fast.w" in keys else []) \
        + [0.0] * (8 if "res.w" in keys else 0)
    old = len(a) > 3 + ap.NR * ap.RES  # the condition before the fix
    print(f"RL3_ATTC {ap.ATTC}: heads {sorted(keys)}  pad action width {len(a)}: randomise attitude: before {old}, now {ap.has_att(keys)}")

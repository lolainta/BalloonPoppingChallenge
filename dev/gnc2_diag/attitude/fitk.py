"""Fit the rotational constant K (dw/dt = K T/m sin g) of a sim run: python fitk.py <variant> <name>"""
import numpy as np, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import quat_to_R
v, n = sys.argv[1], sys.argv[2]
d = np.load(Path(__file__).resolve().parent / f"runs/{v}/{n}.npz")
R = d['R']; A = d['A']; live = ~np.isnan(R[:, 0]); i0 = np.argmax(live)
b = live.copy(); b[:i0 + 5] = False; b[i0 + 2990:] = False; b[-2:] = False
vv = R[:, 3:6]; a = np.full_like(vv, np.nan); a[1:-1] = (vv[2:] - vv[:-2]) / 0.02
Rm = quat_to_R(np.where(live[:, None], R[:, 6:10], [1, 0, 0, 0])); Tm = np.einsum('tji,tj->ti', Rm, a - np.array([0, 0, -9.81]))[:, 2]
dw = np.r_[[[np.nan, np.nan]], (R[1:, 10:12] - R[:-1, 10:12]) / 0.01]
bb = b & (np.abs(A[:, :2]).max(1) > 1)
x = np.r_[Tm[bb] * np.sin(np.radians(A[bb, 0])), Tm[bb] * np.sin(np.radians(A[bb, 1]))]; y = np.r_[dw[bb, 0], dw[bb, 1]]
K = (x @ y) / (x @ x)
wxy = np.linalg.norm(R[b, 10:12], axis=1)
print(v, n, 'K', round(K, 2), 'R2', round(1 - ((y - K * x) ** 2).mean() / y.var(), 3), '|w| p95', np.percentile(wxy, 95).round(2), 'max', wxy.max().round(2), 'pops', len(d['pops']))

"""E: BASE.pt (5-output res + tc learner checkpoint) vs V0.npz (the warm-started learner's first published weights, before any
update) vs AFTER.pt (its checkpoint after the updates): weights finite; old res / tc rows start equal, new ones start at zero
(logstd ln 0.1 = RL3_RES_VSTD default) and move; non-res / non-tc parameters' Adam state carried over (step counts continue), res / tc groups fresh.
python check_e.py BASE.pt V0.npz AFTER.pt"""
import sys

import numpy as np
import torch

B, A = (torch.load(p, map_location="cpu") for p in (sys.argv[1], sys.argv[3]))
V = dict(np.load(sys.argv[2]))
print("base:", {k: tuple(v.shape) for k, v in B["net"].items() if k.startswith(("res", "tc."))}, "steps", B["steps"], "version", B["version"])
print("after:", {k: tuple(v.shape) for k, v in A["net"].items() if k.startswith(("res", "tc."))}, "steps", A["steps"], "version", A["version"])
print("all weights finite:", all(bool(torch.isfinite(v).all()) for v in A["net"].values() if v.is_floating_point()))
# v0 (numpy export: .w = weight.T) vs base: old rows equal, new rows zero
rw, rb, ls, tw, tb = V["res.w"].T, V["res.b"], V["res.logstd"], V["tc.w"].T, V["tc.b"]
bn = B["net"]
print("v0 old rows == base: res.w", np.array_equal(rw[:5], bn["res.weight"].numpy()), " res.b", np.array_equal(rb[:5], bn["res.bias"].numpy()),
      " res_logstd", np.array_equal(ls[:5], bn["res_logstd"].clamp(np.log(0.05), np.log(0.6)).numpy()),
      " tc.w", np.array_equal(tw[:9], bn["tc.weight"].numpy()), " tc.b", np.array_equal(tb[:9], bn["tc.bias"].numpy()),
      " other weights", all(np.array_equal(V[k], B["net"][k].numpy()) for k in ("prior", "lprior")))
print("v0 new rows: res.w/b zero", not rw[5:].any() and not rb[5:].any(), " tc.w/b zero", not tw[9:].any() and not tb[9:].any(),
      " logstd", ls[5:].round(4), "(ln 0.1 = %.4f)" % np.log(0.1), " res.vbound", float(V["res.vbound"]))
an = A["net"]
print("after new rows |max|: res.weight %.3e  res.bias %.3e  res_logstd-ln0.1 %.3e  tc.weight %.3e  tc.bias %.3e" % (
    an["res.weight"][5:].abs().max(), an["res.bias"][5:].abs().max(), (an["res_logstd"][5:] - np.log(0.1)).abs().max(),
    an["tc.weight"][9:].abs().max(), an["tc.bias"][9:].abs().max()))
print("after old rows max |change| vs base: res.weight %.3e  tc.weight %.3e" % (
    (an["res.weight"][:5] - bn["res.weight"]).abs().max(), (an["tc.weight"][:9] - bn["tc.weight"]).abs().max()))
print("base keys all kept:", set(B["net"]) <= set(A["net"]), " new keys:", sorted(set(A["net"]) - set(B["net"])))
gb, ga = B["opt"]["param_groups"], A["opt"]["param_groups"]
print("param groups (sizes) base", [len(g["params"]) for g in gb], "after", [len(g["params"]) for g in ga])
step = lambda o, i: float(o["state"][i]["step"]) if i in o["state"] else -1.0  # noqa: E731  (-1: no state, e.g. frozen lprior)
for gi, g in enumerate(ga):
    print(f"  group {gi}: Adam step counts after {sorted({step(A['opt'], i) for i in g['params']})}"
          + (f", base {sorted({step(B['opt'], i) for i in gb[gi]['params']})}" if gi < len(gb) else " (new group)"))

"""C: BASE.pt (8-output res + tc + tc2 learner checkpoint) vs AFTER.pt (after the RL3_FAST warm start and its updates, or after a
further resume): weights finite; fast head present and moved away from zero; every base key kept; Adam state of the old groups carried
over (step counts continue), the fast group last (fresh on the warm start).   python check_d.py BASE.pt AFTER.pt"""
import sys

import torch

B, A = (torch.load(p, map_location="cpu") for p in sys.argv[1:3])
hd = ("res", "tc.", "tc2.", "fast")
print("base:", {k: tuple(v.shape) for k, v in B["net"].items() if k.startswith(hd)}, "steps", B["steps"], "version", B["version"])
print("after:", {k: tuple(v.shape) for k, v in A["net"].items() if k.startswith(hd)}, "steps", A["steps"], "version", A["version"])
print("all weights finite:", all(bool(torch.isfinite(v).all()) for v in A["net"].values() if v.is_floating_point()))
mx = lambda s, k: float(s["net"][k].abs().max()) if k in s["net"] else None  # noqa: E731
print("fast |w| max %.3e |b| max %.3e  fastc |w| max %.3e |b| max %.3e  fast_logstd %s" % (
    mx(A, "fast.weight"), mx(A, "fast.bias"), mx(A, "fastc.weight"), mx(A, "fastc.bias"), A["net"]["fast_logstd"].numpy().round(4)),
    *(["(base: %.3e / %.3e / %.3e / %.3e)" % tuple(mx(B, k) for k in ("fast.weight", "fast.bias", "fastc.weight", "fastc.bias"))]
      if "fast.weight" in B["net"] else []))
print("base keys all kept:", set(B["net"]) <= set(A["net"]), " new keys:", sorted(set(A["net"]) - set(B["net"])),
      " max |change| of base weights %.3e" % max(float((A["net"][k] - B["net"][k]).abs().max()) for k in B["net"]))
gb, ga = B["opt"]["param_groups"], A["opt"]["param_groups"]
print("param groups (sizes) base", [len(g["params"]) for g in gb], "after", [len(g["params"]) for g in ga])
step = lambda o, i: float(o["state"][i]["step"]) if i in o["state"] else -1.0  # noqa: E731  (-1: no state, e.g. frozen lprior)
for gi, g in enumerate(ga):
    print(f"  group {gi}: Adam step counts after {sorted({step(A['opt'], i) for i in g['params']})}"
          + (f", base {sorted({step(B['opt'], i) for i in gb[gi]['params']})}" if gi < len(gb) else " (new group)"))
if len(gb) <= len(ga):  # old groups: exp_avg of a parameter that kept its index also kept its Adam moments (same shapes)
    same = sum(i in B["opt"]["state"] and i in A["opt"]["state"] and B["opt"]["state"][i]["exp_avg"].shape == A["opt"]["state"][i]["exp_avg"].shape
               for g in gb for i in g["params"])
    print(f"  old-group parameters with Adam moments in both: {same}/{sum(len(g['params']) for g in gb)}")

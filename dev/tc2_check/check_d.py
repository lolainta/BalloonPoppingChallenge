"""D: BASE.pt (8-output res + tc learner checkpoint) vs AFTER.pt (after the RL3_TCOND2 warm start and its updates, or after a
further resume): weights finite; tc2 present and moved away from zero; every base key kept; Adam state of the old groups carried
over (step counts continue), the tc2 group last (fresh on the warm start).   python check_d.py BASE.pt AFTER.pt"""
import sys

import torch

B, A = (torch.load(p, map_location="cpu") for p in sys.argv[1:3])
hd = ("res", "tc.", "tc2.")
print("base:", {k: tuple(v.shape) for k, v in B["net"].items() if k.startswith(hd)}, "steps", B["steps"], "version", B["version"])
print("after:", {k: tuple(v.shape) for k, v in A["net"].items() if k.startswith(hd)}, "steps", A["steps"], "version", A["version"])
print("all weights finite:", all(bool(torch.isfinite(v).all()) for v in A["net"].values() if v.is_floating_point()))
print("tc2 |w| max %.3e  |b| max %.3e" % (A["net"]["tc2.weight"].abs().max(), A["net"]["tc2.bias"].abs().max()),
      *(["(base: %.3e / %.3e)" % (B["net"]["tc2.weight"].abs().max(), B["net"]["tc2.bias"].abs().max())] if "tc2.weight" in B["net"] else []))
print("base keys all kept:", set(B["net"]) <= set(A["net"]), " new keys:", sorted(set(A["net"]) - set(B["net"])),
      " max |change| of base weights %.3e" % max(float((A["net"][k] - B["net"][k]).abs().max()) for k in B["net"]))
gb, ga = B["opt"]["param_groups"], A["opt"]["param_groups"]
print("param groups (sizes) base", [len(g["params"]) for g in gb], "after", [len(g["params"]) for g in ga])
step = lambda o, i: float(o["state"][i]["step"]) if i in o["state"] else -1.0  # noqa: E731  (-1: no state, e.g. frozen lprior)
for gi, g in enumerate(ga):
    print(f"  group {gi}: Adam step counts after {sorted({step(A['opt'], i) for i in g['params']})}"
          + (f", base {sorted({step(B['opt'], i) for i in gb[gi]['params']})}" if gi < len(gb) else " (new group)"))

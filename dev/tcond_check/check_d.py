"""D: compare the async_last.pt saved before the tc warm start (BASE) with the one after it (AFTER):
weights finite, tc moved away from zero, non-tc parameters / optimizer state carried over (Adam step counts continue,
the tc group starts fresh), param-group layout.   python dev/tcond_check/check_d.py BASE.pt AFTER.pt"""
import sys

import torch

B, A = (torch.load(p, map_location="cpu") for p in sys.argv[1:3])
print("base:", {k: v.shape[0] for k, v in B["net"].items() if k.startswith(("res", "tc."))}, "steps", B["steps"], "version", B["version"])
print("after:", {k: tuple(v.shape) for k, v in A["net"].items() if k.startswith(("res", "tc."))}, "steps", A["steps"], "version", A["version"])
print("all weights finite:", all(bool(torch.isfinite(v).all()) for v in A["net"].values() if v.is_floating_point()))
print("tc |w| max %.3e  |b| max %.3e" % (A["net"]["tc.weight"].abs().max(), A["net"]["tc.bias"].abs().max()))
print("base keys all kept:", set(B["net"]) <= set(A["net"]), " new keys:", sorted(set(A["net"]) - set(B["net"])),
      " max |change| of non-tc weights %.3e" % max(float((A["net"][k] - B["net"][k]).abs().max()) for k in B["net"]))
gb, ga = B["opt"]["param_groups"], A["opt"]["param_groups"]
print("param groups (sizes) base", [len(g["params"]) for g in gb], "after", [len(g["params"]) for g in ga])
step = lambda o, i: float(o["state"][i]["step"]) if i in o["state"] else -1.0  # noqa: E731  (-1: no state, e.g. frozen lprior)
for gi, g in enumerate(ga):
    print(f"  group {gi}: Adam step counts after {sorted({step(A['opt'], i) for i in g['params']})}"
          + (f", base {sorted({step(B['opt'], i) for i in gb[gi]['params']})}" if gi < len(gb) else " (new group)"))

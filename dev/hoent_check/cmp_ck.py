"""A: bitwise comparison of two learner checkpoints (net, Adam state, steps / version, temperatures).  cmp_ck.py A.pt B.pt"""
import sys

import torch

a, b = (torch.load(f, map_location="cpu") for f in sys.argv[1:3])
net = [k for k in a["net"] if not torch.equal(a["net"][k], b["net"][k])]
print(f"net tensors {len(a['net'])}, keys equal {a['net'].keys() == b['net'].keys()}, bitwise different: {net or 'none'}")
sa, sb = a["opt"]["state"], b["opt"]["state"]
od = [(i, k) for i in sa for k in sa[i] if not torch.equal(torch.as_tensor(sa[i][k]), torch.as_tensor(sb[i][k]))]
print(f"Adam states {len(sa)} / {len(sb)}, param groups {len(a['opt']['param_groups'])} / {len(b['opt']['param_groups'])}, "
      f"bitwise different: {od or 'none'}")
for k in sorted(set(a) | set(b)):
    if k not in ("net", "opt"):
        print(f"  {k}: {a.get(k, '(absent)')} | {b.get(k, '(absent)')}")
d = max(float((a["net"][k] - b["net"][k]).abs().max()) for k in a["net"])
print("IDENTICAL" if not net and not od and all(a[k] == b[k] for k in ("steps", "version", "log_alpha")) else f"DIFFERENT (max |dw| {d:.3e})")

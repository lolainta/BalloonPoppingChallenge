"""Check 6: 3-output-head checkpoint -> RL3_RES2 learner: rows 0-2 start at the old values, rows 3-4 at zero / ln 0.3 and
then move; weights finite; Adam state of the non-res parameters continued."""
import sys
import numpy as np
import torch

old, new, v0 = sys.argv[1:4]
o, n, P0 = torch.load(old, map_location="cpu"), torch.load(new, map_location="cpu"), dict(np.load(v0))
W0, Wn = o["net"]["res.weight"].numpy(), n["net"]["res.weight"].numpy()
print("res.weight shape old", W0.shape, "new", Wn.shape, "| updates", n["version"] - o["version"], "| all weights finite:",
      all(torch.isfinite(v).all().item() for v in n["net"].values()))
print("start (live_policy before the first update): rows 0-2 == old:", np.array_equal(P0["res.w"][:, :3], W0.T),
      np.array_equal(P0["res.b"][:3], o["net"]["res.bias"].numpy()), np.array_equal(P0["res.logstd"][:3], o["net"]["res_logstd"].numpy()),
      "| rows 3-4 w/b zero:", not P0["res.w"][:, 3:].any() and not P0["res.b"][3:].any(), "logstd", P0["res.logstd"][3:].tolist(),
      "| sbound/nbound", float(P0["res.sbound"]), float(P0["res.nbound"]))
print(f"after: max|w rows 3-4| {np.abs(Wn[3:]).max():.2e}, b rows 3-4 {n['net']['res.bias'][3:].tolist()}, logstd {n['net']['res_logstd'].tolist()}; "
      f"rows 0-2 moved by max {np.abs(Wn[:3] - W0).max():.2e}")
so, sn = o["opt"]["state"], n["opt"]["state"]
print("opt groups old", [len(g["params"]) for g in o["opt"]["param_groups"]], "new", [len(g["params"]) for g in n["opt"]["param_groups"]])
g0 = [i for i in n["opt"]["param_groups"][0]["params"] if i in so]
g1 = n["opt"]["param_groups"][1]["params"]
st = lambda s, i: float(s[i]["step"])
print(f"non-res params with state: {len(g0)}; step old {st(so, g0[0]):.0f} -> new {st(sn, g0[0]):.0f}; all continued "
      f"(new - old == res group step): {all(st(sn, i) - st(so, i) == st(sn, g1[0]) for i in g0)}; res group steps {[st(sn, i) for i in g1]} "
      f"(old res group steps were {[st(so, i) for i in o['opt']['param_groups'][1]['params']]})")

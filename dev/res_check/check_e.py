"""E: after the RL3_RES warm start: finite weights, the res head moved, the old parameters' Adam state was continued."""
import sys
import numpy as np
import torch

old, new, npz = sys.argv[1:4]
o, n = torch.load(old, map_location="cpu"), torch.load(new, map_location="cpu")
print("old groups", [len(g["params"]) for g in o["opt"]["param_groups"]], "new groups", [len(g["params"]) for g in n["opt"]["param_groups"]],
      "updates", n["version"] - o["version"])
print("all weights finite:", all(torch.isfinite(v).all().item() for v in n["net"].values()))
print("res head:", {k: (float(v.abs().max()), v.flatten()[:3].tolist()) for k, v in n["net"].items() if k.startswith("res")})
so, sn = o["opt"]["state"], n["opt"]["state"]
g0 = [i for i in n["opt"]["param_groups"][0]["params"] if i in so]; g1 = n["opt"]["param_groups"][1]["params"]
st = lambda s, i: float(s[i]["step"])
print(f"old params with Adam state: {len(g0)} (without: frozen ones, e.g. lprior under RL3_FREE_LAUNCH); "
      f"same ones after: {sorted(g0) == sorted(i for i in n['opt']['param_groups'][0]['params'] if i in sn)}")
print(f"Adam step: old params before {st(so, g0[0]):.0f} -> after {st(sn, g0[0]):.0f} (all old params continued: "
      f"{all(st(sn, i) - st(so, i) == st(sn, g1[0]) for i in g0)}), res params {[st(sn, i) for i in g1]}")
cos = [float(torch.nn.functional.cosine_similarity(so[i]["exp_avg_sq"].flatten(), sn[i]["exp_avg_sq"].flatten(), 0)) for i in g0]
print(f"old params' exp_avg_sq cosine(before, after): min {min(cos):.3f} median {np.median(cos):.3f}")
P = dict(np.load(npz))
print("exported:", {k: (P[k].shape, float(np.abs(P[k]).max())) if P[k].size > 3 else P[k].tolist() for k in P if k.startswith("res")})

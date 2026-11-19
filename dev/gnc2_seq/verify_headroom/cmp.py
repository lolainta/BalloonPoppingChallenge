import json, sys, numpy as np
tag = sys.argv[1] if len(sys.argv) > 1 else "c20"
F = {json.loads(l)["seed"]: len(json.loads(l)["pops"]) for l in open("dev/gnc2_seq/s4_seq_search/out/free.jsonl")}
S = {json.loads(l)["seed"]: json.loads(l)["best"]["pops"] for l in open("dev/gnc2_seq/s4_seq_search/out/b4k5.jsonl")}
C = {}
for l in open(f"dev/gnc2_seq/verify_headroom/out_{tag}.jsonl"):
    r = json.loads(l); C[r["seed"]] = r
ss = sorted(C)
se = lambda x: "%.3f +- %.3f" % (np.mean(x), np.std(x, ddof=1) / np.sqrt(len(x)))
print("n", len(ss), "greedy==free", sum(C[s]["greedy"] == F[s] for s in ss), "trunc", sum(C[s]["trunc"] for s in ss))
print("chaos gain", se([C[s]["best"] - F[s] for s in ss]), " seq gain same seeds", se([S[s] - F[s] for s in ss]),
      " seq-chaos", se([S[s] - C[s]["best"] for s in ss]))
print([(s, F[s], C[s]["best"], S[s]) for s in ss if C[s]["best"] != F[s] or S[s] != F[s]])

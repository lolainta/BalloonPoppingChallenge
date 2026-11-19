import json, numpy as np
from collections import Counter
O="./dev/gnc2_seq/approach/out/"
def L(f): return [json.loads(l) for l in open(O+f) if l.strip()]
A=L("appr08.jsonl"); C=L("chaos.jsonl"); Fr=L("free.jsonl")
print("errors", [r["seed"] for r in A+C if "error" in r])
print("dupA", [k for k,v in Counter(r["seed"] for r in A).items() if v>1], "dupC",[k for k,v in Counter(r["seed"] for r in C).items() if v>1])
fm=[r for r in Fr if r.get("mode")=="free"]
print("free dup", [k for k,v in Counter(r["seed"] for r in fm).items() if v>1], "free errors", [r["seed"] for r in fm if "error" in r])
pend={r["seed"] for r in Fr if r.get("mode")=="free_pending"}; print("pending without free", sorted(pend-{r['seed'] for r in fm}))
F={r["seed"]:r for r in fm}
A={r["seed"]:r for r in A}; C={r["seed"]:r for r in C}
print("A seeds", sorted(A)); print("C missing vs A", sorted(set(A)-set(C)), "C extra", sorted(set(C)-set(A)))
# params
print(Counter((r["beam"],r["k"],r["tsw"],r["eps"],r["mode"]) for r in list(A.values())+list(C.values())))
def se(x): x=np.array(x,float); return f"{x.mean():+.3f} +- {x.std(ddof=1)/np.sqrt(len(x)):.3f} n={len(x)}"
fp=lambda s: len(F[s]["pops"])
for nm,R in (("A",A),("C",C)):
    ss=sorted(R)
    ex=[s for s in ss if R[s]["greedy"] and R[s]["greedy"]["route"]["pops"]==F[s]["pops"]]
    # also check greedy opts all zero, ranks all zero
    gz=[s for s in ss if R[s]["greedy"] and all(j==0 for j in R[s]["greedy"]["ranks"])]
    print(nm,"exact",len(ex),"/",len(ss),"greedy ranks all0",len(gz),"trunc",sum(R[s]["trunc"] for s in ss),"legs",np.mean([R[s]["legs"] for s in ss]))
    # consistency best pops vs route
    bad=[s for s in ss if R[s]["best"]["pops"]!=len(R[s]["best"]["route"]["pops"])]
    print(" best pops!=route len",bad, " greedy pops vs free len", [s for s in ss if R[s]["greedy"]["pops"]!=fp(s)])
    g=[R[s]["best"]["pops"]-fp(s) for s in ss]
    print(" policy",np.mean([fp(s) for s in ss]),"best",np.mean([R[s]["best"]["pops"] for s in ss]),"gain",se(g),sorted(Counter(g).items()))
    print(" best>=greedy all", all(R[s]["best"]["pops"]>=R[s]["greedy"]["pops"] for s in ss), "max finished==best", all(max(R[s]["finished_pops"])==R[s]["best"]["pops"] for s in ss))
both=sorted(set(A)&set(C))
net=[A[s]["best"]["pops"]-C[s]["best"]["pops"] for s in both]
print("both n",len(both),"appr gain",se([A[s]["best"]["pops"]-fp(s) for s in both]),"chaos gain",se([C[s]["best"]["pops"]-fp(s) for s in both]))
print("net",se(net),sorted(Counter(net).items()))
print("net - via gains diff identical?", np.allclose(net,[ (A[s]["best"]["pops"]-fp(s))-(C[s]["best"]["pops"]-fp(s)) for s in both]))
# burn split
def split(route):
    be=route["burn_end"] or 1e9
    return sum(t<be for t,_ in route["pops"]), sum(t>=be for t,_ in route["pops"])
gb=[split(A[s]["best"]["route"])[0]-split(F[s])[0] for s in sorted(A)]
ga=[split(A[s]["best"]["route"])[1]-split(F[s])[1] for s in sorted(A)]
print("burn gain",se(gb),"post",se(ga))
print("burn_end same free vs best?", sum(abs((A[s]["best"]["route"]["burn_end"] or 0)-(F[s]["burn_end"] or 0))>1e-6 for s in A))
# options
nbp=sum(len(A[s]["best"]["ranks"]) for s in A); OPTS=["none","h+3","h-3","h+6","h-6","v+4","v-4","to2"]
cnt=Counter(OPTS[j] for s in A for j in A[s]["best"]["ranks"]); print("bp",nbp,{k:round(v/nbp*100,1) for k,v in cnt.items()})
nz=[sum(j!=0 for j in A[s]["best"]["ranks"]) for s in A]; print("nz mean",np.mean(nz),sorted(Counter(nz).items()))
gs=[s for s in A if A[s]["best"]["pops"]>fp(s)]; print("gain seeds",len(gs),np.mean([sum(j!=0 for j in A[s]["best"]["ranks"]) for s in gs]),Counter(OPTS[j] for s in gs for j in A[s]["best"]["ranks"] if j))
for s in (46019,46028): print(s, A[s]["best"]["ranks"], fp(s), A[s]["best"]["pops"])
# ranks len vs opts in route
# angles
def ang(route):
    be=route["burn_end"] or np.inf; P=route["pops"]
    return [a for i,a in enumerate(route["ang"]) if a is not None and P[i+1][0]<be]
for nm,get,ss in (("pol",lambda s:F[s],sorted(A)),("appr",lambda s:A[s]["best"]["route"],sorted(A)),("chaos",lambda s:C[s]["best"]["route"],sorted(C))):
    p=np.concatenate([ang(get(s)) for s in ss]); print(nm,np.median(p),p.mean(),len(p))
pd=[np.mean(ang(A[s]["best"]["route"]))-np.mean(ang(F[s])) for s in A if ang(A[s]["best"]["route"]) and ang(F[s])]; print("paired ang",se(pd))
pg=np.concatenate([ang(F[s]) for s in gs]); ag=np.concatenate([ang(A[s]["best"]["route"]) for s in gs]); print("gain seeds med",np.median(pg),np.median(ag))

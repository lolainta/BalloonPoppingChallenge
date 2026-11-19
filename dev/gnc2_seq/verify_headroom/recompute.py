import json, numpy as np
from collections import Counter
O="dev/gnc2_seq/s4_seq_search/out/"
def L(t): 
    rows=[json.loads(l) for l in open(O+t+".jsonl") if l.strip()]
    return rows
free_rows=L("free"); print("free rows",len(free_rows),"distinct",len({r['seed'] for r in free_rows}),"errors",sum('error' in r for r in free_rows))
free={r['seed']:r for r in free_rows}
se=lambda x:(np.mean(x),np.std(x,ddof=1)/np.sqrt(len(x)))
B={}
for t in ["b4k5","b2k3","b8k6"]:
    rows=L(t); seeds=[r['seed'] for r in rows]
    print(t,"rows",len(rows),"distinct",len(set(seeds)),"err",sum('error' in r for r in rows),"range",min(seeds),max(seeds),"trunc",sum(r['trunc'] for r in rows))
    R={r['seed']:r for r in rows}; B[t]=R
    # consistency: best pops == len(route pops); greedy == free
    bad=[s for s,r in R.items() if r['best']['pops']!=len(r['best']['route']['pops'])]
    ex=sum(r['greedy'] is not None and r['greedy']['route']['pops']==free[s]['pops'] for s,r in R.items())
    # any finals with trunc? best trunc?
    print("  best pops!=len(route)",bad," greedy==free",ex,"/",len(R), " best.trunc",sum(r['best']['trunc'] for r in R.values()))
    # max finished_pops equals best?
    print("  best==max finished",all(r['best']['pops']==max(r['finished_pops']) for r in R.values()))
    g=np.array([R[s]['best']['pops']-len(free[s]['pops']) for s in sorted(R)])
    fp=np.array([len(free[s]['pops']) for s in sorted(R)])
    print("  policy %.3f best %.3f gain %.3f +- %.3f"%(fp.mean(),fp.mean()+g.mean(),*se(g)),dict(sorted(Counter(g.tolist()).items())),"median",np.median(g), "neg",sum(g<0))
    # dup balloon ids in best routes?
    d=[s for s,r in R.items() if len({b for _,b in r['best']['route']['pops']})!=len(r['best']['route']['pops'])]
    print("  dup balloons in best route",d)
    # uniq pop times monotone
com=sorted(set(B['b4k5'])&set(B['b2k3']))
d=[B['b4k5'][s]['best']['pops']-B['b2k3'][s]['best']['pops'] for s in com]
print("b4k5-b2k3 %.3f +- %.3f"%se(d),"worse",sum(x<0 for x in d))
u=[max(B['b4k5'][s]['best']['pops'],B['b2k3'][s]['best']['pops'])-len(free[s]['pops']) for s in com]
print("union %.3f +- %.3f"%se(u),dict(sorted(Counter(u).items())))
s8=sorted(B['b8k6'])
for t in B: print(t,"on b8 seeds %.3f +- %.3f"%se([B[t][s]['best']['pops']-len(free[s]['pops']) for s in s8]))
print("all3 %.3f +- %.3f"%se([max(B[t][s]['best']['pops'] for t in B)-len(free[s]['pops']) for s in s8]))
print([(s,len(free[s]['pops']),*(B[t][s]['best']['pops'] for t in B)) for s in s8])
# hop stats, my own
R=B['b4k5']; seeds=sorted(R)
def st(route):
    P=[t for t,_ in route['pops']]; be=route['burn_end'] or 1e9
    nb=sum(t<be for t in P)
    tm=route['tmin'][:max(nb-1,0)]
    return dict(nb=nb,post=len(P)-nb,gaps=np.diff(P[:nb]),tmin=[8 if x is None else min(x,8) for x in tm],tmin_raw=tm,bend=[x for x in route['bend'][:max(nb-2,0)] if x is not None])
F={s:st(free[s]) for s in seeds}; Bs={s:st(R[s]['best']['route']) for s in seeds}
for k in ['gaps','tmin','bend']:
    pf=np.concatenate([F[s][k] for s in seeds]); pb=np.concatenate([Bs[s][k] for s in seeds])
    pm=[np.mean(Bs[s][k])-np.mean(F[s][k]) for s in seeds if len(Bs[s][k]) and len(F[s][k])]
    gs=[s for s in seeds if R[s]['best']['pops']>len(free[s]['pops'])]
    pg=[np.mean(Bs[s][k])-np.mean(F[s][k]) for s in gs if len(Bs[s][k]) and len(F[s][k])]
    print(k,"median pol %.2f best %.2f paired %.3f+-%.3f (n=%d) gained %.3f+-%.3f n=%d"%(np.median(pf),np.median(pb),*se(pm),len(pm),*se(pg),len(pg)))
print("None tmin count", sum(x is None for s in seeds for x in F[s]['tmin_raw']), sum(x is None for s in seeds for x in Bs[s]['tmin_raw']))
print("inburn diff %.3f+-%.3f post %.3f+-%.3f"%(*se([Bs[s]['nb']-F[s]['nb'] for s in seeds]),*se([Bs[s]['post']-F[s]['post'] for s in seeds])))
rk=[R[s]['best']['ranks'] for s in seeds]; print("nontop1/seed",np.mean([sum(j>0 for j in r) for r in rk]),Counter(j for r in rk for j in r), "branchpts/seed",np.mean([len(r) for r in rk]))

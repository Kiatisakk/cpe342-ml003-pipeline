import numpy as np,pandas as pd,json
from sklearn.metrics import roc_auc_score
from det import *
df=pd.read_csv('data/test.csv');Z=prep(build(df))
M=np.load('out/rule_M.npy');names=list(pd.read_csv('out/rule_names.csv').iloc[:,0]);U=M.any(1);K=int(U.sum())
rng=np.random.RandomState(1)
nulls={'col':prep(build(shuffle_cols(df,rng))),'block':prep(build(shuffle_blocks(df,rng)))}
res={}
for n,f in DET.items():
    s=np.load(f'out/{n}_score.npy')
    top=np.argsort(-s)[:K]; prec=U[top].mean()
    r={'AUC_vs_union':roc_auc_score(U,s),'prec@1811':prec}
    for i,g in enumerate(names): r['AUC_'+g[:10]]=roc_auc_score(M[:,i],s)
    for nm,Zn in nulls.items():
        sn=f(Zn,0)
        r[f'null_{nm}_max']=float(sn.max()); r[f'null_{nm}_q999']=float(np.quantile(sn,.999))
        r[f'real_above_null_{nm}_max']=int((s>sn.max()).sum()); r[f'real_above_null_{nm}_q99.9']=int((s>np.quantile(sn,.999)).sum())
        r[f'null_{nm}_above_realq(1-K/N)']=int((sn>np.quantile(s,1-K/len(s))).sum())
    # stability across seeds
    if n in('iforest','ocsvm','gmm'):
        tops=[set(np.argsort(-f(Z,sd))[:K]) for sd in (1,2)]
        r['jaccard_seed1_2']=len(tops[0]&tops[1])/len(tops[0]|tops[1])
        r['jaccard_seed0_1']=len(set(top)&tops[0])/len(set(top)|tops[0])
    # subsample stability: fit on 50% score all not available generically -> use half rows ranking correlation
    idx=rng.choice(len(Z),len(Z)//2,replace=False); ss=f(Z[idx],5)
    from scipy.stats import spearmanr
    r['spearman_full_vs_half']=float(spearmanr(s[idx],ss)[0])
    res[n]=r; print(n,json.dumps({k:round(v,3) if isinstance(v,float) else v for k,v in r.items()}),flush=True)
json.dump(res,open('out/det_eval.json','w'),indent=1)

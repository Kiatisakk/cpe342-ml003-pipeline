import sys,os
sys.path.insert(0,'..')
from common import *
import common; common.DATA=os.path.abspath("../../../cpe342-kaggle-main/01_binary_classification")
import lightgbm as lgb,time
T0=time.time()
lab,unl,te,ids=common.load();y=lab[TARGET].astype(int).values
ref=pd.concat([lab,unl]);X=fe_v2(lab,ref)
rt=lab[SIGNAL].isna().mean().values;re=te[SIGNAL].isna().mean().values
q=np.clip((re-rt)/(1-rt),0,None)   # extra masking to reach test rate (only where test misses more)
print(dict(zip(SIGNAL,q.round(3))))
def mask(df,p,rng):
    d=df.copy();M=rng.random((len(d),len(SIGNAL)))<p[None,:];v=d[SIGNAL].values.copy();v[M]=np.nan;d[SIGNAL]=v;return fe_v2(d,ref)
def run(variant,seed=42,LR=0.08):
    oof=np.zeros(len(y));oofm=np.zeros(len(y))
    NF=int(os.environ.get('NF',5))
    for k,(a,b) in enumerate(folds_for(y,seed)):
        if k>=NF: break
        print('fold',k,time.time()-T0,flush=True)
        rng=np.random.default_rng(seed+k);a=rng.permutation(a);cut=int(.9*len(a));tr,va=a[:cut],a[cut:]
        parts=[X.iloc[tr]];ws=[1.0]
        if variant=='A': ps=[rt,rt]
        elif variant=='B': ps=[q,rt]          # one copy test-shift masking, one train-rate
        elif variant=='C': ps=[q,q]
        elif variant=='N': ps=[]
        for p in ps: parts.append(mask(lab.iloc[tr],p,rng))
        Xa=pd.concat(parts,ignore_index=True);ya=np.tile(y[tr],len(parts))
        # test-like validation: mask va and b with q
        Xvm=mask(lab.iloc[va],q,np.random.default_rng(99+k))
        m=lgb.LGBMClassifier(n_estimators=6000,learning_rate=LR,random_state=seed+k,**LGB_BASE)
        m.fit(Xa,ya,eval_set=[(Xvm,y[va])],eval_metric='auc',callbacks=[lgb.early_stopping(150,verbose=False)])
        oof[b]=m.predict_proba(X.iloc[b])[:,1]
        oofm[b]=m.predict_proba(mask(lab.iloc[b],q,np.random.default_rng(7+k)))[:,1]
    return oof,oofm
if __name__=='__main__':
    v=sys.argv[1];o,om=run(v);NF=int(os.environ.get('NF',5));idx=np.concatenate([b for _,b in folds_for(y,42)[:NF]]);np.save(f'oof_{v}.npy',np.c_[o,om])
    for nm,p in[('natural',o),('testlike',om)]:
        print(v,nm,'AUC',round(roc_auc_score(y[idx],p[idx]),5),'F2b',round(best_thr(y[idx],p[idx],'binary')[1],5),flush=True)

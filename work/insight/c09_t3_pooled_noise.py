# T3: the multiplicative noise has the same shape for every payer (c02-c04). So instead of 11 separate quantile regressions, use ONE location
# (the model's median) and the pooled residual quantile function, estimated on the other folds (nested). Same p, same hurdle rule.
import sys, time, pickle, numpy as np, pandas as pd, lightgbm as lgb, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,"t3/research")
from common import nmae, hurdle_median, extra_feats, prev_offset
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
D=pickle.load(open("t3/research/cache/data_v4.pkl","rb")); Z,_=extra_feats(D["X"],D["Xt"]); y=D["y"]; F=D["folds"]; n=len(y)
ALPHAS=np.array([.02,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5]); loss="MultiQuantile:alpha="+",".join(str(a) for a in ALPHAS)
seg=Z.segment.values.astype(int); p=np.zeros(n); Q=np.zeros((n,len(ALPHAS))); QP=np.zeros((n,len(ALPHAS))); QS=np.zeros((n,len(ALPHAS))); t0=time.time()
for k,(tr,va) in enumerate(F):
    i,j=train_test_split(tr,test_size=.1,random_state=0)
    c=lgb.LGBMClassifier(n_estimators=4000,learning_rate=.03,num_leaves=63,min_child_samples=40,subsample=.8,subsample_freq=1,colsample_bytree=.7,random_state=0,verbose=-1,n_jobs=4)
    c.fit(Z.iloc[i],(y[i]>0).astype(int),eval_set=[(Z.iloc[j],(y[j]>0).astype(int))],callbacks=[lgb.early_stopping(100,verbose=False)]); p[va]=c.predict_proba(Z.iloc[va])[:,1]
    pos=tr[y[tr]>0]; Zp=Z.iloc[pos]; r=np.log(y[pos])-prev_offset(Zp); a,b=train_test_split(np.arange(len(r)),test_size=.1,random_state=0)
    m=CatBoostRegressor(loss_function=loss,iterations=20000,learning_rate=.1,depth=4,l2_leaf_reg=3,od_type="Iter",od_wait=150,random_seed=0,thread_count=4,verbose=False)
    m.fit(Zp.iloc[a],r[a],eval_set=(Zp.iloc[b],r[b])); R=m.predict(Z.iloc[va]); off=prev_offset(Z.iloc[va]); Q[va]=np.exp(R+off[:,None])
    # pooled residual quantiles around the fitted median, from the early-stopping holdout payers of this fold (not used for fitting the trees)
    e=r[b]-m.predict(Zp.iloc[b])[:,-1]; sb=seg[pos][b]
    qa=np.quantile(e,ALPHAS); QP[va]=np.exp(R[:,-1:]+qa[None,:]+off[:,None])
    for s in range(4):
        mk=seg[va]==s; qs=np.quantile(e[sb==s],ALPHAS) if (sb==s).sum()>200 else qa; QS[va[mk]]=np.exp(R[mk,-1:]+qs[None,:]+off[mk,None])
    print(f"fold {k} {time.time()-t0:.0f}s",flush=True)
np.save("work/insight/c09_p.npy",p); np.save("work/insight/c09_Q.npy",Q)
out={"multi-quantile (recipe)":Q,"median + pooled residual quantiles":QP,"median + per-segment residual quantiles":QS,"geo-mean recipe x pooled":np.sqrt(Q*QP),"geo-mean recipe x per-segment":np.sqrt(Q*QS)}
base=hurdle_median(p,Q,ALPHAS)
for nm,q in out.items():
    pr=hurdle_median(p,q,ALPHAS); d=[nmae(y[b],pr[b])-nmae(y[b],base[b]) for _,b in F]
    print(f"{nm:40s} OOF {nmae(y,pr):.5f}  delta per fold {np.round(d,5)}  | by segment err ratio",{s:round(np.abs(y-pr)[seg==s].sum()/y[seg==s].sum(),4) for s in range(4)})

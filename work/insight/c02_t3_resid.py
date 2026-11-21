# T3: how much of log(y/prev) is explained, what the residual looks like, and whether the whale spike tail is predictable.
import pandas as pd, numpy as np, lightgbm as lgb, warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from scipy.stats import skew, kurtosis
tr=pd.read_csv("cpe342-kaggle-main/03_regression/train.csv"); y=tr.spending_30d
print("y at cap 300000:",(y==300000).sum(),"| y>250k:",(y>250000).sum(),"| by segment at cap:",tr[y==300000].segment.fillna(-1).value_counts().to_dict())
F=[c for c in tr.columns if c not in ("player_id","spending_30d")]
pos=tr[(y>0)&tr.prev_month_spending.notna()&tr.segment.notna()].reset_index(drop=True); pos["lr"]=np.log(pos.spending_30d/pos.prev_month_spending)
oof=np.zeros(len(pos))
for a,b in KFold(5,shuffle=True,random_state=0).split(pos):
    m=lgb.LGBMRegressor(n_estimators=600,learning_rate=.03,num_leaves=15,min_child_samples=50,subsample=.8,subsample_freq=1,colsample_bytree=.8,objective="l1",verbose=-1).fit(pos.loc[a,F],pos.lr[a])
    oof[b]=m.predict(pos.loc[b,F])
pos["res"]=pos.lr-oof
print("\nresidual of log r after LGB-L1 (payers, known segment & prev)")
print(pos.groupby("segment").res.agg(["size","median","std",lambda v:np.subtract(*np.quantile(v,[.75,.25]))/1.349,skew,kurtosis,lambda v:np.mean(v>1),lambda v:np.mean(v<-1)]).round(4).set_axis(["n","median","sd","robust_sd","skew","kurt","res>1","res<-1"],axis=1))
print("raw lr sd per segment:",pos.groupby("segment").lr.std().round(3).to_dict())
# tail: whales with residual > 0.8 (spike)
w=pos[pos.segment==3].reset_index(drop=True); t=(w.res>0.8).astype(int); print("\nwhale spike rows (res>0.8):",t.sum(),"of",len(w), "| share at cap among them %.3f"%(w.spending_30d[t==1]==300000).mean())
print("spike y / prev quantiles:",np.quantile(np.exp(w.lr[t==1]),[.05,.25,.5,.75,.95]).round(1), "| y quantiles",np.quantile(w.spending_30d[t==1],[.05,.5,.95]).round(0))
p=np.zeros(len(w))
for a,b in KFold(5,shuffle=True,random_state=0).split(w):
    p[b]=lgb.LGBMClassifier(n_estimators=300,learning_rate=.03,num_leaves=15,min_child_samples=30,verbose=-1).fit(w.loc[a,F],t[a]).predict_proba(w.loc[b,F])[:,1]
print("spike predictability AUC %.4f"%roc_auc_score(t,p))
print("univariate AUC of spike vs each feature:"); print(pd.Series({c:roc_auc_score(t[w[c].notna()],w[c].dropna()) for c in F}).sort_values().round(3).iloc[[0,1,2,3,-4,-3,-2,-1]].to_string())
# is the residual scale predictable? (|res| model)
ab=np.abs(pos.res); q=np.zeros(len(pos))
for a,b in KFold(5,shuffle=True,random_state=1).split(pos):
    q[b]=lgb.LGBMRegressor(n_estimators=300,learning_rate=.03,num_leaves=15,min_child_samples=50,verbose=-1).fit(pos.loc[a,F],ab[a]).predict(pos.loc[b,F])
print("\n|res| predictability: corr(pred,|res|) %.4f ; by segment mean|res|"%np.corrcoef(q,ab)[0,1], pos.groupby("segment").res.apply(lambda v:np.abs(v).mean()).round(4).to_dict())
for s in range(4):
    mk=pos.segment==s; print(f"  seg {s}: corr(pred,|res|) within segment %.4f"%np.corrcoef(q[mk],ab[mk])[0,1])
pos[["player_id","segment","lr","res"]].to_csv("work/insight/c02_resid.csv",index=False)

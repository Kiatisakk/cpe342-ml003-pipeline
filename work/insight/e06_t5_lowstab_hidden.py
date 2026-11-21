# T5 NEW: can hidden lowstab rows (behavior_stability NaN) be told apart from normal rows that pass the same age / purchase_frequency box?
# Positives = visible lowstab rows; negatives = normal rows with behaviour_stability observed that pass the same box on the other metrics.
import pandas as pd, numpy as np, lightgbm as lgb, warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
X=pd.read_csv("cpe342-kaggle-main/05_anomaly_detection/test.csv"); G=pd.read_csv("work/t5v2/out/t5_groups.csv"); P=[1,2,3,4]
pred=pd.read_csv("t5/preds/v2data/t5v2_union_box_lsnan.csv").iloc[:,1].values==1; post=pd.read_csv("work/t5v2/out/lowstab_nan_scores.csv").post_lowstab_nan.values
anyg=G.iloc[:,1:].max(axis=1).values==1
pf=X[[f"purchase_frequency_{p}" for p in P]].values; age=X.account_age_days.values; bs=X.behavior_stability.values
box=(((pf>=3)&(pf<=6))|np.isnan(pf)).all(1)&(((age>=800)&(age<=1500))|np.isnan(age))
acc=np.ones(len(X),bool)
for b,(lo,hi) in {"performance_stability":(.85,.95),"kda_std":(.7,1.0),"accuracy_std":(9,12)}.items():
    m=X[[f"{b}_{p}" for p in P]].values; acc&=(((m>=lo)&(m<=hi))|np.isnan(m)).all(1)
pos=box&(bs>=.2)&(bs<=.4); neg=box&~np.isnan(bs)&~pos&~anyg&~acc; cand=box&np.isnan(bs)&~anyg&~acc
print("visible lowstab",pos.sum(),"| normal rows passing the same box",neg.sum(),"| candidates (bs NaN)",cand.sum())
base=sorted({c[:-2] for c in X.columns if c[-2:] in ("_1","_2","_3","_4")}); F={}
for b in base:
    m=X[[f"{b}_{p}" for p in P]]; F[b+"_mean"]=m.mean(axis=1); F[b+"_sd"]=m.std(axis=1); F[b+"_min"]=m.min(axis=1); F[b+"_max"]=m.max(axis=1)
F=pd.DataFrame(F); 
for c in X.columns:
    if c not in ("player_id","behavior_stability") and c[-2:] not in ("_1","_2","_3","_4"): F[c]=X[c]
F["n_nan"]=X.iloc[:,1:].isna().sum(axis=1)
tr=pos|neg; y=pos[tr].astype(int); A=F[tr].reset_index(drop=True)
ua=pd.Series({c:roc_auc_score(y[A[c].notna()],A[c].dropna()) for c in A.columns}); ua=(ua-.5).abs().sort_values(ascending=False)
print("\ntop univariate |AUC-0.5| (positives vs box-passing normals):"); print(ua.head(14).round(3).to_string())
o=np.zeros(len(A)); sc=np.zeros(cand.sum())
for s in range(3):
    for a,b in StratifiedKFold(5,shuffle=True,random_state=s).split(A,y):
        m=lgb.LGBMClassifier(n_estimators=300,learning_rate=.03,num_leaves=8,min_child_samples=10,subsample=.8,subsample_freq=1,colsample_bytree=.5,verbose=-1).fit(A.iloc[a],y[a])
        o[b]+=m.predict_proba(A.iloc[b])[:,1]/3; sc+=m.predict_proba(F[cand])[:,1]/15
print("\nCV AUC %.4f"%roc_auc_score(y,o)); 
q=np.quantile(o[y==0],[.5,.9,.99,.999]); print("negatives' OOF score quantiles 50/90/99/99.9%:",q.round(4),"| positives' 1/10/50%:",np.quantile(o[y==1],[.01,.1,.5]).round(4))
ci=np.where(cand)[0]; od=np.argsort(-sc)
print("\ncandidates sorted by classifier score: score | old posterior | in current prediction")
for j in od[:40]: print(f"  row {ci[j]:5d} score {sc[j]:.3f} post {post[ci[j]]:.2f} pred {int(pred[ci[j]])}")
print("candidates with score>0.5:",(sc>.5).sum(),"| >0.2:",(sc>.2).sum(),"| >0.05:",(sc>.05).sum(),"| sum of scores %.1f"%sc.sum())
pd.DataFrame({"row":ci,"clf":sc,"post":post[ci],"pred":pred[ci].astype(int)}).to_csv("work/insight/e06_lowstab_candidates.csv",index=False)

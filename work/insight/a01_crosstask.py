# Cross-task linkage by player_id: do tasks share players, and is one task's label/feature informative for another's?
import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
D="cpe342-kaggle-main/"
T={"t1":"01_binary_classification","t2":"02_multiclass_classification","t3":"03_regression","t5":"05_anomaly_detection"}
X={}
for k,d in T.items():
    for s in ("train","test"):
        try: X[k,s]=pd.read_csv(f"{D}{d}/{s}.csv")
        except FileNotFoundError: pass
for (k,s),df in X.items():
    n=df.player_id.str[1:].astype(int); print(k,s,"id range",n.min(),n.max(),"unique",n.nunique(),"of",len(n))
ks=list(X)
print("\nid overlap counts")
for i,a in enumerate(ks):
    for b in ks[i+1:]:
        print(a,b,len(set(X[a].player_id)&set(X[b].player_id)))
# joins
t1=X["t1","train"].set_index("player_id"); t2=X["t2","train"].set_index("player_id"); t3=pd.concat([X["t3","train"],X["t3","test"]]).set_index("player_id")
j=t2[["segment"]].join(t3[["segment","spending_30d"]],lsuffix="_t2",rsuffix="_t3",how="inner")
print("\nT2.segment x T3.segment (same id):",len(j)); print(pd.crosstab(j.segment_t2,j.segment_t3.fillna(-1)))
j=t1[["is_cheater"]].join(t2[["segment"]],how="inner").dropna(); print("\nT1 cheater rate by T2 segment",len(j)); print(j.groupby("segment").is_cheater.agg(["mean","size"]))
j=t1[["is_cheater"]].join(t3[["segment","spending_30d"]],how="inner"); print("\nT1 cheater by T3 segment"); print(j.groupby(j.segment.fillna(-1)).is_cheater.agg(["mean","size"]))
# shared feature names with equal values?
for a,b in (("t2","t3"),("t1","t2"),("t1","t3")):
    A=pd.concat([X[a,"train"],X[a,"test"]]).set_index("player_id"); B=pd.concat([X[b,"train"],X[b,"test"]]).set_index("player_id")
    for c in set(A.columns)&set(B.columns):
        jj=A[[c]].join(B[[c]],lsuffix="_a",rsuffix="_b",how="inner").dropna()
        try: print(a,b,c,"n",len(jj),"corr",round(np.corrcoef(jj.iloc[:,0].astype(float),jj.iloc[:,1].astype(float))[0,1],4),"equal",round((jj.iloc[:,0]==jj.iloc[:,1]).mean(),4))
        except Exception as e: print(a,b,c,"equal",round((jj.iloc[:,0]==jj.iloc[:,1]).mean(),4))

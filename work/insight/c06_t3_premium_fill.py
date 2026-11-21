# T3: new exact identity is_premium_member = (vip_status >= 3). Does filling the missing side help? Proxy test with LightGBM:
# payer log-ratio residual (L1) and pay/no-pay log-loss, same folds, with and without the fill.
import pandas as pd, numpy as np, lightgbm as lgb, warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss, roc_auc_score
tr=pd.read_csv("cpe342-kaggle-main/03_regression/train.csv"); y=tr.spending_30d.values
F=[c for c in tr.columns if c not in ("player_id","spending_30d")]
A=tr[F].copy(); B=tr[F].copy()
m1=B.is_premium_member.isna()&B.vip_status.notna(); B.loc[m1,"is_premium_member"]=(B.vip_status[m1]>=3).astype(float)
m2=B.vip_status.isna()&B.is_premium_member.notna(); B["vip_lo"]=B.vip_status.fillna(B.is_premium_member*3); B["vip_hi"]=B.vip_status.fillna(2+3*B.is_premium_member)
print("premium filled from vip:",int(m1.sum()),"rows | vip bounded from premium:",int(m2.sum()),"rows")
pay=(y>0).astype(int); pos=np.where((y>0)&tr.prev_month_spending.notna().values)[0]; lr=np.log(y[pos]/tr.prev_month_spending.values[pos])
for nm,X in (("raw",A),("with identity fill",B)):
    o=np.zeros(len(pos)); p=np.zeros(len(y))
    for s in (0,1):
        for a,b in KFold(5,shuffle=True,random_state=s).split(pos):
            o[b]+=lgb.LGBMRegressor(n_estimators=800,learning_rate=.03,max_depth=3,num_leaves=8,min_child_samples=30,objective="l1",verbose=-1).fit(X.iloc[pos[a]],lr[a]).predict(X.iloc[pos[b]])/2
        for a,b in KFold(5,shuffle=True,random_state=s).split(y):
            p[b]+=lgb.LGBMClassifier(n_estimators=400,learning_rate=.03,num_leaves=31,min_child_samples=50,subsample=.8,subsample_freq=1,colsample_bytree=.8,verbose=-1).fit(X.iloc[a],pay[a]).predict_proba(X.iloc[b])[:,1]/2
    r=np.abs(lr-o); k1=m1.values[pos]; k2=m2.values[pos]
    print(f"{nm:20s} payer mean|res| all {r.mean():.5f} | premium-missing rows {r[k1].mean():.5f} | vip-missing rows {r[k2].mean():.5f} || pay log-loss all {log_loss(pay,p):.5f} AUC {roc_auc_score(pay,p):.5f} | premium-missing {log_loss(pay[m1],p[m1]):.5f} | vip-missing {log_loss(pay[m2],p[m2]):.5f}")

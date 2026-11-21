# T3: predict every column from all the others (train features only, target excluded) to find hidden identities for missing-value fills.
import pandas as pd, numpy as np, lightgbm as lgb, warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
tr=pd.read_csv("cpe342-kaggle-main/03_regression/train.csv"); F=[c for c in tr.columns if c not in ("player_id","spending_30d")]
X=tr[F].copy()
for c in ["historical_spending","prev_month_spending","avg_transaction_value","total_playtime_hours"]: X["log_"+c]=np.log1p(X[c])
rows=[]
for c in F:
    d=X[X[c].notna()]; d=d.sample(min(len(d),40000),random_state=0); cols=[k for k in X.columns if k!=c and k!="log_"+c]
    a,b=train_test_split(np.arange(len(d)),test_size=.3,random_state=0); disc=d[c].nunique()<=12
    if disc:
        m=lgb.LGBMClassifier(n_estimators=150,learning_rate=.08,num_leaves=15,verbose=-1).fit(d.iloc[a][cols],d.iloc[a][c].astype(int)); p=m.predict(d.iloc[b][cols])
        s=accuracy_score(d.iloc[b][c].astype(int),p); base=d[c].value_counts(normalize=True).max(); kind="acc"
    else:
        m=lgb.LGBMRegressor(n_estimators=150,learning_rate=.08,num_leaves=15,verbose=-1).fit(d.iloc[a][cols],d.iloc[a][c]); p=m.predict(d.iloc[b][cols]); s=r2_score(d.iloc[b][c],p); base=0; kind="R2"
    imp=pd.Series(m.booster_.feature_importance("gain"),index=cols).sort_values(ascending=False); rows.append((c,round(tr[c].isna().mean(),3),kind,round(s,3),round(base,3),", ".join(imp.index[:3])))
print(pd.DataFrame(rows,columns=["column","NaN","metric","score","majority","top predictors"]).to_string(index=False))

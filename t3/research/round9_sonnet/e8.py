import pandas as pd,numpy as np,lightgbm as lgb
from sklearn.model_selection import cross_val_predict
tr=pd.read_csv('cpe342-kaggle-main/03_regression/train.csv')
P=tr[(tr.spending_30d>0)&tr.prev_month_spending.notna()].copy()
P['r']=np.log(P.spending_30d/P.prev_month_spending)
X=P.drop(columns=['player_id','spending_30d','r'])
for s in [0,1,2,3]:
    S=P[P.segment==s];XS=X.loc[S.index]
    m=lgb.LGBMRegressor(n_estimators=300,learning_rate=0.03,num_leaves=15,verbose=-1,n_jobs=4)
    p=cross_val_predict(m,XS,S.r,cv=5)
    res=S.r-p;print(s,'r std',S.r.std().round(3),'res std',res.std().round(3),'MAD*1.48',(np.median(abs(res-res.median()))*1.48).round(3))
    print(np.percentile(res,[1,5,25,50,75,95,99]).round(2))
    m.fit(XS,S.r);print(pd.Series(m.booster_.feature_importance('gain'),XS.columns).sort_values(ascending=False).head(6).round(0).to_dict())

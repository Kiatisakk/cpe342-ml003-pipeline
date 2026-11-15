import pandas as pd,numpy as np,lightgbm as lgb
from sklearn.model_selection import cross_val_predict
tr=pd.read_csv('cpe342-kaggle-main/03_regression/train.csv')
P=tr[(tr.spending_30d>0)&tr.prev_month_spending.notna()&tr.historical_spending.notna()&(tr.segment<3)].copy()
lp=np.log(P.prev_month_spending);lh=np.log(P.historical_spending)
P['r']=np.log(P.spending_30d)-lp
P['d']=lh-lp;P['m']=(lh+lp)/2
base=['days_since_last_purchase','vip_status','is_premium_member','seasonal_spending_pattern','games_played','purchase_frequency','segment','event_participation_rate','owns_limited_edition','platform','total_transactions']
for name,cols in (('base',base),('+d',base+['d']),('+d,lp,lh',base+['d','m'])):
    X=P[cols].copy()
    if 'm' in cols: X['lp']=lp
    m=lgb.LGBMRegressor(n_estimators=300,learning_rate=0.03,num_leaves=15,verbose=-1,n_jobs=4)
    p=cross_val_predict(m,X,P.r,cv=5);print(name,(P.r-p).std().round(4))
print(P[['r','d','m']].corr().round(3))

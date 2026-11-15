import pandas as pd,numpy as np
tr=pd.read_csv('cpe342-kaggle-main/03_regression/train.csv')
tr['pay']=tr.spending_30d>0
for c in ['days_since_last_purchase','event_participation_rate','purchase_frequency','vip_status','segment','is_premium_member','owns_limited_edition','platform','tournament_participation']:
    q=tr[c].nunique()
    g=pd.qcut(tr[c],10,duplicates='drop') if q>12 else tr[c]
    print(c);print(tr.groupby(g,observed=True).pay.agg(['mean','size']).round(3).T.to_string())
print(tr.groupby(['segment','vip_status']).pay.mean().unstack().round(2))
print(tr.groupby(['segment','is_premium_member']).pay.mean().unstack().round(2))

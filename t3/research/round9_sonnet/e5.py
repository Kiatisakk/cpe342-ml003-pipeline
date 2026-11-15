import pandas as pd,numpy as np
tr=pd.read_csv('cpe342-kaggle-main/03_regression/train.csv')
F=tr.drop(columns=['player_id','spending_30d']);tr['nn']=F.isna().sum(1);tr['pay']=tr.spending_30d>0
print(tr.groupby('nn').agg(n=('pay','size'),pay=('pay','mean'),mean=('spending_30d','mean')).T.round(3).to_string())
print('expected nn dist binom', F.isna().mean().mean()*F.shape[1])
for c in F.columns:
    m=F[c].isna(); 
    if abs(tr.pay[m].mean()-tr.pay[~m].mean())>0.015: print(c,tr.pay[m].mean(),tr.pay[~m].mean())
# which cols jointly missing?
print(F.isna().corr().values[np.triu_indices(F.shape[1],1)].max())
# rounding of features
for c in ['prev_month_spending','historical_spending','avg_transaction_value','event_participation_rate','days_since_last_purchase']:
    print(c,(F[c].dropna()*100%1).abs().lt(1e-6).mean())

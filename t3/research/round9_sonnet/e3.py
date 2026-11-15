import pandas as pd,numpy as np
from sklearn.linear_model import LinearRegression
tr=pd.read_csv('cpe342-kaggle-main/03_regression/train.csv')
P=tr[tr.spending_30d>0].copy()
P['ly']=np.log(P.spending_30d)
for c in ['prev_month_spending','historical_spending','avg_transaction_value','total_transactions']: P['l'+c]=np.log(P[c])
print(P[['ly','lprev_month_spending','lhistorical_spending','lavg_transaction_value','ltotal_transactions']].corr().round(3))
print('prev vs hist corr', P.lprev_month_spending.corr(P.lhistorical_spending))
# ratio prev/hist structure
r=P.lprev_month_spending-P.lhistorical_spending
print(r.describe().round(3))
print('y/prev by segment');print(P.assign(r=P.ly-P.lprev_month_spending).groupby('segment').r.describe().round(3))
print(P.assign(r=P.ly-P.lprev_month_spending).groupby('vip_status').r.mean())
Z=P.dropna(subset=['lprev_month_spending','lhistorical_spending'])
for cols in (['lprev_month_spending'],['lprev_month_spending','lhistorical_spending'],['lprev_month_spending','lhistorical_spending','event_participation_rate','vip_status','is_premium_member','segment']):
    Zz=Z.dropna(subset=cols);m=LinearRegression().fit(Zz[cols],Zz.ly);res=Zz.ly-m.predict(Zz[cols]);print(cols,m.coef_.round(3),res.std().round(3))
res=Z.ly-LinearRegression().fit(Z[['lprev_month_spending','lhistorical_spending']],Z.ly).predict(Z[['lprev_month_spending','lhistorical_spending']])
from scipy import stats
print('skew',stats.skew(res),'kurt',stats.kurtosis(res))
Z=Z.assign(res=res)
print(Z[[c for c in tr.columns if c not in('player_id',)]+['res']].corr(method='spearman')['res'].round(3).sort_values().to_string())

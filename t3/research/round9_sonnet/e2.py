import pandas as pd,numpy as np
tr=pd.read_csv('cpe342-kaggle-main/03_regression/train.csv')
y=tr.spending_30d; print('cap',(y>=300000).sum(),(y==300000).sum(), 'min pos',y[y>0].min())
tr['pay']=(y>0).astype(int)
num=tr.drop(columns=['player_id']).columns
print(tr[num].corr(method='spearman')['pay'].drop(['pay','spending_30d']).sort_values().round(3).to_string())
P=tr[y>0].copy(); P['ly']=np.log(P.spending_30d)
print(P[list(num)+["ly"]].corr(method='spearman')['ly'].drop(['ly','spending_30d','pay']).sort_values().round(3).to_string())
for c in ['prev_month_spending','historical_spending']:
    r=np.log(P.spending_30d/P[c]); print(c,r.describe().round(3).to_dict())
print(tr.groupby('segment').apply(lambda g:pd.Series({'zero':(g.spending_30d==0).mean(),'med':g.spending_30d[g.spending_30d>0].median(),'prevzero':(g.prev_month_spending==0).mean()})))
print('prev==0 vs pay',pd.crosstab(tr.prev_month_spending==0,tr.pay))
print('hist==0',pd.crosstab(tr.historical_spending==0,tr.pay))

import pandas as pd,numpy as np
tr=pd.read_csv('cpe342-kaggle-main/03_regression/train.csv');te=pd.read_csv('cpe342-kaggle-main/03_regression/test.csv')
for d in (tr,te): d['n']=d.player_id.str[1:].astype(int)
print(tr.n.min(),tr.n.max(),te.n.min(),te.n.max())
print('dups',tr.player_id.duplicated().sum(),te.player_id.duplicated().sum(),te.player_id.isin(tr.player_id).sum())
b=pd.cut(tr.n,[-1,51999,84499,110499,10**7],labels=False)
print(pd.crosstab(b,tr.segment,dropna=False))
print(tr.groupby(pd.qcut(tr.n,20)).apply(lambda g:pd.Series({'zero':(g.spending_30d==0).mean(),'mean':g.spending_30d.mean(),'seg':g.segment.mean()})))
print(tr.n.corr(tr.spending_30d,method='spearman'), (tr.n%2).groupby(tr.spending_30d>0).mean())
print(tr.n.diff().abs().describe())
y=tr.spending_30d;p=y[y>0]
print(p.describe());print((p*100%1).abs().describe()); print(np.round(p.head(10),6).tolist())

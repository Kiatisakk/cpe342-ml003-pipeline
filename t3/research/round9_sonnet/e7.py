import pandas as pd,numpy as np
from scipy import stats
tr=pd.read_csv('cpe342-kaggle-main/03_regression/train.csv')
o=pd.read_csv('t3/research/best_oof.csv');print(o.columns.tolist(),len(o))
o['seg']=tr.segment.values;o['prev']=tr.prev_month_spending.values;o['pay']=o.y>0
P=o[o.pay&(o.pred>0)]
e=np.log(P.y/P.pred);print('payers w/ pred>0 log err',e.describe().round(3).to_dict())
print(P.assign(e=e).groupby('seg').e.agg(['std','mean']))
# error mass
err=(o.y-o.pred).abs();print('err share: zero-y rows',err[~o.pay].sum()/err.sum(),'payers',err[o.pay].sum()/err.sum())
print('err share y>1e5',err[o.y>1e5].sum()/err.sum(), 'pred0 payers',err[o.pay&(o.pred==0)].sum()/err.sum())

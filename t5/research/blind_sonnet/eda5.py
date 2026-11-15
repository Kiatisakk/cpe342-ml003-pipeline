import warnings;warnings.filterwarnings('ignore')
from feat import *
cols={}
for m in metrics:
    if m in('login_lat','login_lon'):continue
    X=d[[f'{m}_{i}' for i in range(1,4)]].values   # periods 1-3 only (p4 may be contaminated)
    cols[m+'_mean123']=np.nanmean(X,1)
    cols[m+'_std123']=np.nanstd(X,1)
for c in ['account_age_days','behavior_stability']: cols[c]=d[c].values
for k,v in cols.items():
    v=v[~np.isnan(v)]
    h,_=np.histogram(v,bins=30)
    print(f'{k:42s}',' '.join(map(str,h)))

import warnings;warnings.filterwarnings('ignore')
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score
d=pd.read_csv('data/test.csv')
M=np.load('out/rule_M.npy');names=list(pd.read_csv('out/rule_names.csv').iloc[:,0])
U=M.any(1)
fl=['password_changed','vpn_usage','suspicious_login_time','mass_item_sale']
t=pd.DataFrame({g:d[fl][M[:,i]].mean() for i,g in enumerate(names)}); t['normal']=d[fl][~U].mean(); print(t.round(3))
print('account_age by group:',{g:round(d.account_age_days[M[:,i]].mean()) for i,g in enumerate(names)},'normal',round(d.account_age_days[~U].mean()))
# rows with NaN in group-defining features
print('NaN behavior_stability',d.behavior_stability.isna().sum(),' expected missed lowbeh ~',round(d.behavior_stability.isna().sum()*274/ d.behavior_stability.notna().sum()))
# is lowbeh predictable from other cols? AUC of each col mean
low=(d.behavior_stability<0.55)
best=[]
for c in d.columns[1:]:
    v=d[c];ok=v.notna()&d.behavior_stability.notna()
    if v[ok].nunique()>1: best.append((abs(roc_auc_score(low[ok],v[ok])-.5),c))
print(sorted(best)[-5:])
# per-group NaN rate
print('NaN rate per row in groups', {g:round(d.drop(columns='player_id')[M[:,i]].isna().mean().mean(),4) for i,g in enumerate(names)},'normal',round(d.drop(columns='player_id')[~U].isna().mean().mean(),4))

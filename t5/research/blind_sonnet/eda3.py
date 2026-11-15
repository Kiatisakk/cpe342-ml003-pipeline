import warnings;warnings.filterwarnings('ignore')
from feat import *
for m in ['accuracy_mean','kda_mean','skill_rating','skill_trajectory','play_pattern_consistency','reaction_time_mean','location_changes','friend_removes']:
    X=d[[f'{m}_{i}' for i in range(1,5)]].values
    base=np.nanmean(X[:,:3],1)
    r=X[:,3]/base if m not in('skill_trajectory',) else X[:,3]-base
    r=r[~np.isnan(r)]
    q=np.percentile(r,[0.1,1,5,50,95,99,99.9])
    print(m,np.round(q,2))
    h,e=np.histogram(r,bins=40); print(h)

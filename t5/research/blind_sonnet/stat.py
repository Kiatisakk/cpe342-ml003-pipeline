import warnings;warnings.filterwarnings('ignore')
from feat import *
def m123(m): return np.nanmean(d[[f'{m}_{i}' for i in range(1,4)]].values,1)
F={'ipent':m123('ip_hash_entropy')<0.25,'ltv':m123('login_time_variance')<1.0,'sessstd':m123('session_duration_std')<6.5,
 'kdastd':m123('kda_std')<0.146,'comm':m123('communication_frequency')<0.17,'act':m123('activity_level')<0.34,'beh':d.behavior_stability.values<0.55}
F=pd.DataFrame(F).fillna(False)
print(F.sum())
print(F.astype(int).T.dot(F.astype(int)))
print('any',F.any(axis=1).sum(),' nflags dist',F.sum(1).value_counts().to_dict())
F.to_pickle('out/stat_flags.pkl')

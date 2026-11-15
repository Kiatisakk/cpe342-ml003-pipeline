import pandas as pd,numpy as np,lightgbm as lgb
from sklearn.model_selection import cross_val_predict,StratifiedKFold
from sklearn.metrics import roc_auc_score
D='cpe342-kaggle-main/01_binary_classification/'
a=pd.read_csv(D+'train.csv');b=pd.read_csv(D+'test.csv')
fe=[c for c in a.columns if c not in('player_id','is_cheater')]
Z=pd.concat([a[fe],b[fe]]);y=np.r_[np.zeros(len(a)),np.ones(len(b))]
cv=StratifiedKFold(5,shuffle=True,random_state=0)
m=lgb.LGBMClassifier(n_estimators=200,learning_rate=0.05,num_leaves=15,verbose=-1)
r={c:roc_auc_score(y,cross_val_predict(m,Z[[c]],y,cv=cv,method='predict_proba')[:,1]) for c in fe}
print(sorted(r.items(),key=lambda x:-x[1])[:8])
for c in ['reports_received','friend_network_size','device_changes_count']:
    print(c,a[c].describe().round(2).to_dict(),b[c].describe().round(2).to_dict())

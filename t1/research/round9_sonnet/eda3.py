import pandas as pd,numpy as np,lightgbm as lgb
from sklearn.model_selection import cross_val_predict,StratifiedKFold
from sklearn.metrics import roc_auc_score
D='cpe342-kaggle-main/01_binary_classification/'
a=pd.read_csv(D+'train.csv');b=pd.read_csv(D+'test.csv')
for d in(a,b): d['n']=d.player_id.str[1:].astype(int)
l=a[a.is_cheater.notna()]
cv=StratifiedKFold(5,shuffle=True,random_state=0)
m=lgb.LGBMClassifier(n_estimators=200,learning_rate=0.03,num_leaves=15,verbose=-1)
# id-only features
X=pd.DataFrame({'n':l.n,'m2':l.n%2,'m10':l.n%10,'m100':l.n%100,'m1000':l.n%1000,'m7':l.n%7,'m13':l.n%13})
p=cross_val_predict(m,X,l.is_cheater,cv=cv,method='predict_proba')[:,1];print('id-only AUC',roc_auc_score(l.is_cheater,p))
# adversarial
fe=[c for c in a.columns if c not in('player_id','n','is_cheater')]
Z=pd.concat([a[fe].assign(n=a.n),b[fe].assign(n=b.n)]);y=np.r_[np.zeros(len(a)),np.ones(len(b))]
p=cross_val_predict(m,Z[fe],y,cv=cv,method='predict_proba')[:,1];print('adv AUC feats',roc_auc_score(y,p))
p=cross_val_predict(m,Z[['n']],y,cv=cv,method='predict_proba')[:,1];print('adv AUC id',roc_auc_score(y,p))
t=pd.read_csv('t1/research/t1_best_test.csv');print(t.pred.mean(),t.score.describe())

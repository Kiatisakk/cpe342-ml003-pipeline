import pandas as pd,numpy as np,lightgbm as lgb
from sklearn.model_selection import cross_val_predict
tr=pd.read_csv('cpe342-kaggle-main/03_regression/train.csv')
K=tr[tr.segment.notna()]
X=K.drop(columns=['player_id','segment','spending_30d'])
m=lgb.LGBMClassifier(n_estimators=150,learning_rate=0.08,verbose=-1,n_jobs=4)
p=cross_val_predict(m,X,K.segment,cv=5,method='predict_proba')
print('acc',(p.argmax(1)==K.segment).mean())
m.fit(X,K.segment);print(pd.Series(m.booster_.feature_importance('gain'),X.columns).sort_values(ascending=False).head(10).round(0))
X2=X.drop(columns=['spending_30d'],errors='ignore')
# with target
X3=X.assign(y=K.spending_30d);p=cross_val_predict(m,X3,K.segment,cv=5,method='predict_proba');print('acc with y',(p.argmax(1)==K.segment).mean())

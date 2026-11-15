from common import *
import lightgbm as lgb
def fp(Xa,ya,Xs):
    m=lgb.LGBMClassifier(n_estimators=400,learning_rate=0.04,num_leaves=15,min_child_samples=50,subsample=0.8,subsample_freq=1,colsample_bytree=0.7,n_jobs=2,verbose=-1)
    m.fit(Xa,ya);return [m.predict_proba(X) for X in Xs]
for w in (False,True):
    A,B=prep_gbm(w);cv(fp,'lgb_all_'+('B' if w else 'A'),A,B,save=True)
m=lgb.LGBMClassifier(n_estimators=300,n_jobs=2,verbose=-1).fit(*prep_gbm(False)[:1],y)
A=prep_gbm(False)[0];print(sorted(zip(m.feature_importances_,A.columns))[::-1][:25])

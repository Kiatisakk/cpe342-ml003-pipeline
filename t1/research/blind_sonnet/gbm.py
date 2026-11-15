from common import *
import lightgbm as lgb,sys
def mk(params):
    def fp(Xa,ya,Xbs):
        m=lgb.LGBMClassifier(**params,n_jobs=2,verbose=-1).fit(Xa,ya)
        return [m.predict_proba(x)[:,1] for x in Xbs]
    return fp
P=dict(n_estimators=400,learning_rate=0.03,num_leaves=31,min_child_samples=50,subsample=.8,subsample_freq=1,colsample_bytree=.8,reg_lambda=5)
for nm,cols,eng in [('gbm_raw_all',ALL,False),('gbm_raw_sig',SIG,False),('gbm_eng_sig',SIG,True)]:
    run(nm,mk(P),feats(tr,cols,eng).values,feats(te,cols,eng).values,save=(nm=='gbm_eng_sig'))

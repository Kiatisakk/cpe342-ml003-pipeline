from ref import *
import lightgbm as lgb
qs=[0.02,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.98]
def hurdle(Xa,ya,Xb,Xt):
    pos=ya>0
    w=lambda Z:Z.prev_month_spending.fillna(Xa.prev_month_spending.median()).values+1
    c=lgb.LGBMClassifier(n_estimators=400,learning_rate=0.04,num_leaves=31,min_child_samples=40,subsample=0.8,subsample_freq=1,colsample_bytree=0.7,n_jobs=2,verbose=-1).fit(Xa,pos)
    r=ya[pos]/w(Xa)[pos];Xp=Xa[pos]
    ms=[lgb.LGBMRegressor(objective='quantile',alpha=q,n_estimators=300,learning_rate=0.05,num_leaves=31,min_child_samples=40,subsample=0.8,subsample_freq=1,colsample_bytree=0.7,n_jobs=2,verbose=-1).fit(Xp,r) for q in qs]
    def f(Z):
        p=c.predict_proba(Z)[:,1];Q=np.sort(np.column_stack([m.predict(Z) for m in ms]),axis=1)
        tq=np.clip(1-1/(2*np.maximum(p,1e-6)),0,1)
        out=np.array([np.interp(t,qs,Q[i]) if t>0 else 0 for i,t in enumerate(tq)])
        return np.where(p<0.5,0,np.maximum(out,0)*w(Z))
    return f(Xb),f(Xt)
if __name__=='__main__':
    o,t=run('hurdle_lgb_quantile',hurdle);np.save('out/hurdle_lgb_quantile_oof.npy',o);np.save('out/hurdle_lgb_quantile_test.npy',t)

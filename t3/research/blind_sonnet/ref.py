from common import *
import lightgbm as lgb
X=feats(tr);Xt=feats(te)
def run(name,fit):
    t=time.time();oof=np.zeros(len(tr));tp=np.zeros(len(te));s=[]
    for a,b in folds:
        pb,pt=fit(X.iloc[a],y[a],X.iloc[b],Xt)
        oof[b]=pb;tp+=pt/5;s.append(score(y[b],pb))
    print(name,np.mean(s),np.std(s),time.time()-t,flush=True);return oof,tp
def l1(Xa,ya,Xb,Xt):
    m=lgb.LGBMRegressor(objective='l1',n_estimators=800,learning_rate=0.03,num_leaves=63,min_child_samples=40,subsample=0.8,subsample_freq=1,colsample_bytree=0.7,n_jobs=2,verbose=-1).fit(Xa,ya)
    return m.predict(Xb),m.predict(Xt)
def ratio(Xa,ya,Xb,Xt):
    w=Xa.prev_month_spending.fillna(Xa.prev_month_spending.median()).values
    m=lgb.LGBMRegressor(objective='l1',n_estimators=800,learning_rate=0.03,num_leaves=63,min_child_samples=40,subsample=0.8,subsample_freq=1,colsample_bytree=0.7,n_jobs=2,verbose=-1).fit(Xa,ya/ (w+1),sample_weight=w+1)
    f=lambda Z:m.predict(Z)*(Z.prev_month_spending.fillna(Xa.prev_month_spending.median()).values+1)
    return f(Xb),f(Xt)
if __name__=='__main__':
    for n,f in [('lgb_l1',l1),('lgb_ratio_wl1',ratio)]:
        o,t=run(n,f);np.save(f'out/{n}_oof.npy',o);np.save(f'out/{n}_test.npy',t)

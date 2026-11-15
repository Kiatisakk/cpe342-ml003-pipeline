from common import *
import sys
from sklearn.preprocessing import StandardScaler,QuantileTransformer
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDRegressor,QuantileRegressor
from sklearn.svm import LinearSVR,SVR
X=feats(tr);Xt=feats(te)
cols=list(X.columns)
def prep(Xa,Xb,Xc):
    med=Xa.median()
    def f(Z):
        Z=Z.copy();ind=Z[['segment','vip_status','event_participation_rate']].isna().astype(float).add_suffix('_na')
        return pd.concat([Z.fillna(med),ind],axis=1)
    A,B,C=f(Xa),f(Xb),f(Xc)
    for c in ['historical_spending','prev_month_spending','total_transactions','avg_transaction_value','pm_hs','pm_tx','hs_age']:
        for Z in (A,B,C): Z[c]=np.log1p(Z[c].clip(lower=0))
    sc=StandardScaler().fit(A);return sc.transform(A),sc.transform(B),sc.transform(C)
def run(name,fit,sub=None):
    t=time.time();oof=np.zeros(len(tr));tp=np.zeros(len(te));s=[]
    for a,b in folds:
        pb,pt=fit(X.iloc[a],y[a],X.iloc[b],Xt)
        oof[b]=pb;tp+=pt/5;s.append(score(y[b],pb))
    print(name,round(np.mean(s),5),round(np.std(s),5),round(time.time()-t),flush=True)
    np.save(f'out/{name}_oof.npy',oof);np.save(f'out/{name}_test.npy',tp)
# target: ratio y/(pm+1) with weight pm+1 -> absolute loss on y equivalently. Linear-SVR eps=0 ~ LAD
def mk(kind):
    def fit(Xa,ya,Xb,Xt):
        pa,pb,pt=prep(Xa,Xb,Xc=Xt)
        if kind.startswith('nys'):
            ns=Nystroem(gamma=float(kind.split('_')[1]),n_components=int(kind.split('_')[2]),random_state=0,n_jobs=1).fit(pa[:20000])
            pa,pb,pt=[np.hstack([q,ns.transform(q)]) for q in (pa,pb,pt)]
        w=Xa.prev_month_spending.fillna(Xa.prev_month_spending.median()).values+1
        # lin eps-insens loss on ratio target with sample weight
        m=SGDRegressor(loss='epsilon_insensitive',epsilon=0.0,alpha=1e-6,penalty='l2',max_iter=60,tol=None,learning_rate='adaptive',eta0=0.01,random_state=0)
        r=np.clip(ya/w,0,20)
        # weighted by w (scaled)
        m.fit(pa,r,sample_weight=w/w.mean())
        g=lambda P,Z:np.maximum(m.predict(P),0)*(Z.prev_month_spending.fillna(Xa.prev_month_spending.median()).values+1)
        return g(pb,Xb),g(pt,Xt)
    return fit
if __name__=='__main__':
    pass
    run('svr_nys_0.02_600',mk('nys_0.02_600'))

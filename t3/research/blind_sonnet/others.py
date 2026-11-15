from ref import *
from svr import prep
import sys
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import ExtraTreesRegressor
def knn(Xa,ya,Xb,Xt):
    pa,pb,pt=prep(Xa,Xb,Xt)
    keep=[i for i,c in enumerate(list(Xa.columns)) if c in ('lpm','lhs','segment','vip_status','event_participation_rate','is_premium_member','total_transactions','avg_transaction_value','purchases_on_discount','days_since_last_purchase')]
    pa,pb,pt=pa[:,keep],pb[:,keep],pt[:,keep]
    w=Xa.prev_month_spending.fillna(Xa.prev_month_spending.median()).values+1
    # median of ratio via neighbors: use k=50 mean-> use median manually
    m=KNeighborsRegressor(n_neighbors=40,n_jobs=2).fit(pa,np.arange(len(pa)))
    def f(P,Z):
        idx=m.kneighbors(P,return_distance=False);rr=(ya/w)[idx]
        # weighted median of y/w with weights w
        ww=w[idx];o=np.argsort(rr,1);rs=np.take_along_axis(rr,o,1);ws=np.take_along_axis(ww,o,1);cs=np.cumsum(ws,1)/ws.sum(1,keepdims=True)
        med=rs[np.arange(len(rs)),(cs>=0.5).argmax(1)]
        return med*(Z.prev_month_spending.fillna(Xa.prev_month_spending.median()).values+1)
    return f(pb,Xb),f(pt,Xt)
def et(Xa,ya,Xb,Xt):
    med=Xa.median();w=Xa.prev_month_spending.fillna(med.prev_month_spending).values+1
    m=ExtraTreesRegressor(200,min_samples_leaf=10,max_features=0.6,n_jobs=2,random_state=0).fit(Xa.fillna(-1),np.clip(ya/w,0,30),sample_weight=w)  # weighted-mean (not median) of ratio
    f=lambda Z:m.predict(Z.fillna(-1))*(Z.prev_month_spending.fillna(med.prev_month_spending).values+1)
    return f(Xb),f(Xt)
if __name__=='__main__':
    for n,f in [('knn_wmedian',knn),('extratrees_ratio',et)][int(sys.argv[1]):int(sys.argv[1])+1]:
        o,t=run(n,f);np.save(f'out/{n}_oof.npy',o);np.save(f'out/{n}_test.npy',t)

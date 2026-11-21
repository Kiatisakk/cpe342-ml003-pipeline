# T3 A/B on the real recipe (one seed, CatBoost lr 0.1 for speed): cached v4 features vs the same with the new identity
# is_premium_member = (vip_status >= 3) used as a fill (premium from vip; vip lower/upper bounds from premium).
import sys, time, pickle, numpy as np, pandas as pd, lightgbm as lgb, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,"t3/research")
from common import nmae, hurdle_median, extra_feats, prev_offset
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
D=pickle.load(open("t3/research/cache/data_v4.pkl","rb")); X,_=extra_feats(D["X"],D["Xt"]); y=D["y"]; F=D["folds"]; n=len(y)
ALPHAS=np.array([.02,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5]); loss="MultiQuantile:alpha="+",".join(str(a) for a in ALPHAS)
def variant(fill):
    Z=X.copy()
    if fill:
        m=Z.is_premium_member.isna()&Z.vip_status.notna(); Z.loc[m,"is_premium_member"]=(Z.vip_status[m]>=3).astype(float)
        Z["vip_lo"]=Z.vip_status.fillna(Z.is_premium_member*3); Z["vip_hi"]=Z.vip_status.fillna(2+3*Z.is_premium_member)
    return Z
res={}
for nm,fill in (("base",False),("fill",True)):
    Z=variant(fill); p=np.zeros(n); Q=np.zeros((n,len(ALPHAS))); t0=time.time()
    for k,(tr,va) in enumerate(F):
        i,j=train_test_split(tr,test_size=.1,random_state=0)
        c=lgb.LGBMClassifier(n_estimators=4000,learning_rate=.03,num_leaves=63,min_child_samples=40,subsample=.8,subsample_freq=1,colsample_bytree=.7,random_state=0,verbose=-1,n_jobs=4)
        c.fit(Z.iloc[i],(y[i]>0).astype(int),eval_set=[(Z.iloc[j],(y[j]>0).astype(int))],callbacks=[lgb.early_stopping(100,verbose=False)]); p[va]=c.predict_proba(Z.iloc[va])[:,1]
        pos=tr[y[tr]>0]; Zp=Z.iloc[pos]; r=np.log(y[pos])-prev_offset(Zp); a,b=train_test_split(np.arange(len(r)),test_size=.1,random_state=0)
        m=CatBoostRegressor(loss_function=loss,iterations=20000,learning_rate=.1,depth=4,l2_leaf_reg=3,od_type="Iter",od_wait=150,random_seed=0,thread_count=4,verbose=False)
        m.fit(Zp.iloc[a],r[a],eval_set=(Zp.iloc[b],r[b])); Q[va]=np.exp(m.predict(Z.iloc[va])+prev_offset(Z.iloc[va])[:,None])
        print(f"  {nm} fold {k} done {time.time()-t0:.0f}s iters {m.get_best_iteration()}",flush=True)
    pred=hurdle_median(p,Q,ALPHAS); res[nm]=pred; fs=[nmae(y[b],pred[b]) for _,b in F]
    print(f"{nm}: OOF {nmae(y,pred):.5f} | folds {np.round(fs,5)}",flush=True); np.save(f"work/insight/c08_{nm}_oof.npy",pred)
d=[nmae(y[b],res['fill'][b])-nmae(y[b],res['base'][b]) for _,b in F]; print("fill - base: OOF %+.5f | per fold"%(nmae(y,res['fill'])-nmae(y,res['base'])),np.round(d,5))

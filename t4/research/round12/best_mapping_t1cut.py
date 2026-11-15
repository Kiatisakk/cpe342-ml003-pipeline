import numpy as np,pandas as pd,itertools
# --- T4: all 120 mappings under the chance model, two share fits
C=['7-11k','fiveN','freefried','pube','roblock']; S=pd.read_csv('t4/research/t4_test_str.csv').pred
a=S.value_counts(normalize=True).reindex(C).values
fits={'w.2':np.array([.1353,.2988,.2398,.2106,.1154]),'wfree':np.array([.1358,.3024,.2104,.2014,.15])}
def mac(m,s):
    return np.mean([2*a[i]*s[m[i]]/(a[i]+s[m[i]]) for i in range(5)])
r=pd.DataFrame([dict(map=' '.join(map(str,m)),a=mac(m,fits['w.2']),b=mac(m,fits['wfree'])) for m in itertools.permutations(range(5))])
r['avg']=(r.a+r.b)/2; r=r.sort_values('avg',ascending=False); print('range macro',r.a.min().round(4),r.a.max().round(4))
print(r.head(6).round(4).to_string()); 
for k in ['0 1 2 3 4','0 3 2 1 4','0 1 4 3 2','3 1 0 2 4','4 1 2 0 3']: print(r[r['map']==k].round(4).to_string(header=False))
# --- T1: fine cut grid
D="cpe342-kaggle-main/01_binary_classification/"
tr=pd.read_csv(D+"train.csv"); te=pd.read_csv(D+"test.csv"); l=tr[tr.is_cheater.notna()].reset_index(drop=True); y=l.is_cheater.astype(int).values
O=np.load("t1/research/best_oof_avg3.npy"); sc=pd.read_csv("t1/research/t1_best_test.csv").score.values
key=["account_age_days","friend_network_size","reports_received","aiming_smoothness","crosshair_placement"]
ct,cv=l[key].notna().all(axis=1).values,te[key].notna().all(axis=1).mean(); w=np.where(ct,cv/ct.mean(),(1-cv)/(1-ct.mean()))
def F2(p,yy,ww):
    tp=(ww*(p&(yy==1))).sum();fp=(ww*(p&(yy==0))).sum();fn=(ww*(~p&(yy==1))).sum();return 5*tp/(5*tp+4*fn+fp)
cur=np.quantile(sc,.40); print('current test cut',round(cur,4))
g=np.arange(.15,.241,.005); rng=np.random.default_rng(0); B=[rng.integers(0,len(y),len(y)) for _ in range(300)]
print('cut testrate F2plain F2weighted | boot gain vs current mean sd P(gain>0)')
base=[F2(O[i]>=cur,y[i],w[i]) for i in B]
for t in g:
    d=np.array([F2(O[i]>=t,y[i],w[i]) for i in B])-base
    print(round(t,3),round((sc>=t).mean(),4),round(F2(O>=t,y,np.ones(len(y))),5),round(F2(O>=t,y,w),5),'|',round(d.mean(),5),round(d.std(),5),round((d>0).mean(),2))
wins=np.array([g[np.argmax([F2(O[i]>=t,y[i],w[i]) for t in g])] for i in B]); print('bootstrap best cut median',np.median(wins),'iqr',np.percentile(wins,[25,75]))

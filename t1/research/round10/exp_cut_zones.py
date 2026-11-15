import numpy as np, pandas as pd, itertools
from sklearn.model_selection import StratifiedKFold
D="cpe342-kaggle-main/01_binary_classification/"
tr=pd.read_csv(D+"train.csv"); te=pd.read_csv(D+"test.csv"); l=tr[tr.is_cheater.notna()].reset_index(drop=True); y=l.is_cheater.astype(int).values
O=np.load("t1/research/best_oof_avg3.npy"); sc=pd.read_csv("t1/research/t1_best_test.csv").score.values
def F2(p,yy=y,w=None):
    w=np.ones(len(yy)) if w is None else w
    tp=(w*(p&(yy==1))).sum();fp=(w*(p&(yy==0))).sum();fn=(w*(~p&(yy==1))).sum();return 5*tp/(5*tp+4*fn+fp)
key=["account_age_days","friend_network_size","reports_received","aiming_smoothness","crosshair_placement"]
ct,cv=l[key].notna().all(axis=1).values,te[key].notna().all(axis=1).mean(); w=np.where(ct,cv/ct.mean(),(1-cv)/(1-ct.mean()))
print("test complete",round(cv,3),"train",round(ct.mean(),3))
print("cut | test rate | OOF F2 plain | OOF F2 test-like weights")
for t in [.145,.16,.17,.18,.188,.2,.2267,.25,.3]:
    print(t,round((sc>=t).mean(),4),round(F2(O>=t),5),round(F2(O>=t,w=w),5))
kf=StratifiedKFold(5,shuffle=True,random_state=7); grid=np.arange(.12,.32,.004); r=[];ch=[]
for a,b in kf.split(O,y):
    t=grid[int(np.argmax([F2(O[a]>=g,y[a],w[a]) for g in grid]))]; ch.append(round(float(t),3))
    r.append(F2(O[b]>=t,y[b],w[b])-F2(O[b]>=.2267,y[b],w[b]))
print("nested test-like cut",ch,"gain vs p.2267",round(np.mean(r),5),"+-",round(np.std(r),5))
v=pd.read_csv("submissions/FINAL/sub_v16a_LEAK_T1r60_T4intDatacard_T5idblock.csv",dtype=str).task1.values=="1.0"; print("current rate",round(v.mean(),4),"diff vs .188",int((v!=(sc>=.188)).sum()))
cols=["account_age_days","friend_network_size","reports_received"]; B=24
def rects(xa,ya,ca,cls,minn):
    e=[np.unique(np.nanquantile(xa[:,i],np.linspace(0,1,B+1))) for i in range(2)]
    ix=np.clip(np.searchsorted(e[0],xa[:,0],side="right")-1,0,len(e[0])-2); iy=np.clip(np.searchsorted(e[1],xa[:,1],side="right")-1,0,len(e[1])-2)
    nx,ny=len(e[0])-1,len(e[1])-1; N=np.zeros((nx+1,ny+1)); C=np.zeros((nx+1,ny+1))
    np.add.at(N,(ix+1,iy+1),1); np.add.at(C,(ix+1,iy+1),(ca==cls)*1.); N=N.cumsum(0).cumsum(1); C=C.cumsum(0).cumsum(1)
    out=[]
    for x1,x2 in itertools.combinations(range(nx+1),2):
        n=N[x2,:]-N[x1,:]; c=C[x2,:]-C[x1,:]
        Y1,Y2=np.triu_indices(ny+1,1); nn=n[Y2]-n[Y1]; cc=c[Y2]-c[Y1]; ok=np.where((cc==0)&(nn>=minn))[0]
        if len(ok): j=ok[np.argmax(nn[ok])]; out.append((nn[j],e[0][x1],e[0][x2],e[1][Y1[j]],e[1][Y2[j]]))
    return sorted(out,key=lambda z:-z[0])[:5]
def inside(Z,R):
    m=np.zeros(len(Z),bool)
    for _,a,b,c,d in R: m|=(Z[:,0]>=a)&(Z[:,0]<b)&(Z[:,1]>=c)&(Z[:,1]<d)
    return m
base=O>=np.quantile(O,.40); bt=v
for minn in [30,100,300]:
  for cls,name,new in [(1,"legit-only (flip 1 to 0)",False),(0,"cheater-only (flip 0 to 1)",True)]:
    G=[];n=0;right=0
    for a,b in kf.split(O,y):
        pb=base.copy()
        for c1,c2 in itertools.combinations(cols,2):
            Z=l[[c1,c2]].values; ok=~np.isnan(Z).any(1); aa=a[ok[a]]; R=rects(Z[aa],y[aa],y[aa],cls,minn)
            M=inside(np.nan_to_num(Z,nan=-1e18),R)&ok; h=np.zeros(len(y),bool);h[b]=True;h&=M&(pb!=new)
            pb[h]=new; n+=h.sum(); right+=(y[h]==int(new)).sum()
        G.append(F2(pb[b],y[b])-F2(base[b],y[b]))
    print("minn",minn,name,"flips",n,"correct",right,"rate",round(right/max(n,1),3),"F2 gain",round(np.mean(G),5),"+-",round(np.std(G),5),flush=True)

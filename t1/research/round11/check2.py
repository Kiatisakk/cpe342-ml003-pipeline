import numpy as np,pandas as pd
from sklearn.model_selection import StratifiedKFold
D="cpe342-kaggle-main/01_binary_classification/"
tr=pd.read_csv(D+"train.csv");te=pd.read_csv(D+"test.csv");l=tr[tr.is_cheater.notna()].reset_index(drop=True);y=l.is_cheater.astype(int).values
O=np.load("t1/research/best_oof_avg3.npy")
def F2(p,yy):
    tp=(p&(yy==1)).sum();fp=(p&(yy==0)).sum();fn=(~p&(yy==1)).sum();return 5*tp/(5*tp+4*fn+fp)
cols=["account_age_days","friend_network_size","reports_received","level"]
def ext(Z):
    Z=Z.copy();L,A,S=Z.level,Z.account_age_days,Z.level_progression_speed
    Z["age_lo"]=L*.15/S;Z["age_hi"]=L*1.08/S;Z["lv_lo"]=S*A/1.08;Z["lv_hi"]=S*A/.15;return Z
def rules(a,Zt,yt,lps):
    c=a[yt==1];g=a[yt==0];m0=np.zeros(len(Zt),bool);m1=m0.copy()
    for k in cols:
        v=Zt[k].values;m0|=(v<c[k].min())|(v>c[k].max());m1|=(v<g[k].min())|(v>g[k].max())
    if lps:
        for gg,mm in ((c,0),(g,1)):
            q=(Zt.age_lo.values>gg.account_age_days.max())|(Zt.age_hi.values<gg.account_age_days.min())|(Zt.lv_lo.values>gg.level.max())|(Zt.lv_hi.values<gg.level.min())
            if mm==0:m0|=q
            else:m1|=q
    return m0,m1
X=ext(l);base=O>=.188;kf=StratifiedKFold(5,shuffle=True,random_state=7);Xt=ext(te)
p=pd.read_csv("submissions/FINAL/sub_v16a_LEAK_T1r60_T4intDatacard_T5idblock.csv",dtype=str).task1.values=="1.0"
for lps in (False,True):
    G=[];n0=c0=n1=c1=0
    for a,b in kf.split(O,y):
        m0,m1=rules(X.iloc[a],X.iloc[b],y[a],lps);pb=base[b].copy();h0=m0&pb&~m1;h1=m1&~pb&~m0
        n0+=h0.sum();c0+=(y[b][h0]==0).sum();n1+=h1.sum();c1+=(y[b][h1]==1).sum();pb[h0]=False;pb[h1]=True;G.append(F2(pb,y[b])-F2(base[b],y[b]))
    m0,m1=rules(X,Xt,y,lps)
    print("lps",lps,"flip1to0",n0,"correct",c0,"flip0to1",n1,"correct",c1,"gain",round(np.mean(G),5),"+-",round(np.std(G),5),"| test would flip 1to0",(m0&p&~m1).sum(),"0to1",(m1&~p&~m0).sum(),"conflict",(m0&m1).sum())

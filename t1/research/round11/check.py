import numpy as np,pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
D="cpe342-kaggle-main/01_binary_classification/"
tr=pd.read_csv(D+"train.csv");te=pd.read_csv(D+"test.csv");l=tr[tr.is_cheater.notna()].reset_index(drop=True);y=l.is_cheater.astype(int).values
O=np.load("t1/research/best_oof_avg3.npy")
def F2(p,yy):
    tp=(p&(yy==1)).sum();fp=(p&(yy==0)).sum();fn=(~p&(yy==1)).sum();return 5*tp/(5*tp+4*fn+fp)
cols=[c for c in l.columns if c not in("player_id","is_cheater")]
def ext(Z):
    Z=Z.copy();L,A,S=Z.level,Z.account_age_days,Z.level_progression_speed
    Z["age_lo"]=L*.15/S;Z["age_hi"]=L*1.08/S;Z["lv_lo"]=S*A/1.08;Z["lv_hi"]=S*A/.15;return Z
def rules(a,Zt,yt):
    c=a[yt==1];g=a[yt==0];m0=np.zeros(len(Zt),bool);m1=m0.copy()
    for k in cols:
        v=Zt[k].values
        m0|=(v<c[k].min())|(v>c[k].max());m1|=(v<g[k].min())|(v>g[k].max())
    for gg,mm in ((c,"m0"),(g,"m1")):
        hi,lo=gg.account_age_days.max(),gg.account_age_days.min();lvh,lvl=gg.level.max(),gg.level.min()
        q=(Zt.age_lo.values>hi)|(Zt.age_hi.values<lo)|(Zt.lv_lo.values>lvh)|(Zt.lv_hi.values<lvl)
        if mm=="m0":m0|=q
        else:m1|=q
    return m0,m1
X=ext(l);base=O>=.188;kf=StratifiedKFold(5,shuffle=True,random_state=7);G=[];n0=c0=n1=c1=0
for a,b in kf.split(O,y):
    m0,m1=rules(X.iloc[a],X.iloc[b],y[a]);pb=base[b].copy()
    h0=m0&pb&~m1;h1=m1&~pb&~m0;n0+=h0.sum();c0+=(y[b][h0]==0).sum();n1+=h1.sum();c1+=(y[b][h1]==1).sum()
    pb[h0]=False;pb[h1]=True;G.append(F2(pb,y[b])-F2(base[b],y[b]))
print("T1 nested rules: flip1to0",n0,"correct",c0,"flip0to1",n1,"correct",c1,"F2 gain",round(np.mean(G),5),"+-",round(np.std(G),5))
Xt=ext(te);m0,m1=rules(X,Xt,y);v=pd.read_csv("submissions/FINAL/sub_v16a_LEAK_T1r60_T4intDatacard_T5idblock.csv",dtype=str);p=v.task1.values=="1.0"
print("T1 test: must0",m0.sum(),"must1",m1.sum(),"conflict",(m0&m1).sum(),"would flip 1to0",(m0&p&~m1).sum(),"0to1",(m1&~p&~m0).sum())
np.save("t1/research/round11/test_m0.npy",m0&~m1);np.save("t1/research/round11/test_m1.npy",m1&~m0)
# T2 affine + blend, nested
y2=pd.read_csv("cpe342-kaggle-main/02_multiclass_classification/train.csv").segment.values;P=np.load("t2/research/best_oof_proba.npy")
B="t2/research/blind_sonnet/";Lr=np.load(B+"logreg_spline_A_oof.npy");M=np.load(B+"mlp_torch_A_oof.npy");Bl=.87*P+.08*Lr+.05*M
f=lambda p,yy:f1_score(yy,p,average="macro")
def fp(Pa,ya,it=8):
    pi=np.bincount(ya,minlength=4)/len(ya);q=pi.copy();F=np.full(4,.75);a=np.ones(4);b=np.zeros(4)
    for _ in range(it):
        a=2/(pi+q);b=-F/(pi+q);pr=(a*Pa+b).argmax(1);q=np.bincount(pr,minlength=4)/len(ya);F=np.array([2*((pr==k)&(ya==k)).mean()/(pi[k]+q[k]) for k in range(4)])
    return a,b
print("T2 plain",round(f(P.argmax(1),y2),5),"blend fixed weights",round(f(Bl.argmax(1),y2),5))
res={k:[] for k in("affine","blend","blend+affine")}
for s in (1,2,3):
  for a_,b_ in StratifiedKFold(5,shuffle=True,random_state=s).split(P,y2):
    base=f(P[b_].argmax(1),y2[b_]);A,Bb=fp(P[a_],y2[a_]);res["affine"].append(f((A*P[b_]+Bb).argmax(1),y2[b_])-base)
    res["blend"].append(f(Bl[b_].argmax(1),y2[b_])-base);A,Bb=fp(Bl[a_],y2[a_]);res["blend+affine"].append(f((A*Bl[b_]+Bb).argmax(1),y2[b_])-base)
for k,v_ in res.items():print("T2",k,"gain",round(np.mean(v_),5),"+-",round(np.std(v_),5),"pos folds",sum(x>0 for x in v_),"/",len(v_))
# T3 50/50
a=pd.read_csv("t3/research/best_oof.csv");r=pd.read_csv("t3/research/robust_oof.csv");ta=pd.read_csv("t3/research/t3_best_test.csv");tb=pd.read_csv("t3/research/t3_robust_test.csv")
sc=lambda p,yy:1/(1+np.abs(p-yy).mean()/np.abs(yy).mean());Y=a.y.values;mix=(a.pred.values+r.pred.values)/2
print("T3 best",round(sc(a.pred.values,Y),5),"robust",round(sc(r.pred.values,Y),5),"50/50",round(sc(mix,Y),5),"corr",round(np.corrcoef(a.pred,r.pred)[0,1],5))
d=[sc(mix[b],Y[b])-sc(a.pred.values[b],Y[b]) for _,b in StratifiedKFold(5,shuffle=True,random_state=7).split(Y,Y>0)];print("T3 50/50 paired",np.round(d,5),"mean",round(np.mean(d),5))
print("T3 test mean abs diff",round(float(np.abs(ta.pred-tb.pred).mean()),2),"zero-share",round(float((ta.pred==0).mean()),4),round(float((tb.pred==0).mean()),4))

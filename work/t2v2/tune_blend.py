# T2 new data: nested-CV re-tune of the blend weights (NB / spline LR / MLP) with the affine F1 rule fitted inside.
import numpy as np, pandas as pd, itertools
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
y=pd.read_csv('cpe342-kaggle-main/02_multiclass_classification/train.csv').segment.values; N=len(y)
S='work/t2v2/blind/out/'
O=[np.load('t2/preds/v2data/nb_oof.npy'),np.load(S+'logreg_spline_A_oof.npy'),np.load(S+'mlp_torch_A_oof.npy')]
def affine(Bo,yy):
    pi=np.bincount(yy,minlength=4)/len(yy);q=pi.copy();Fk=np.full(4,.75)
    for _ in range(8):
        a=2/(pi+q);b=-Fk/(pi+q);pr=(a*Bo+b).argmax(1);q=np.bincount(pr,minlength=4)/len(yy)
        Fk=np.array([2*((pr==k)&(yy==k)).mean()/(pi[k]+q[k]) for k in range(4)])
    return a,b
f1=lambda yy,p:f1_score(yy,p,average='macro')
g=np.round(np.arange(0,1.0001,0.05),2); W=[w for w in itertools.product(g,repeat=3) if abs(sum(w)-1)<1e-9 and w[0]>=0.5]
def score(w,idx_fit,idx_ev):
    B=sum(wi*o for wi,o in zip(w,O)); a,b=affine(B[idx_fit],y[idx_fit]); return f1(y[idx_ev],(a*B[idx_ev]+b).argmax(1))
res=[]
for seed in (0,1,2):
    for fit,ev in StratifiedKFold(5,shuffle=True,random_state=seed).split(O[0],y):
        sc=[score(w,fit,fit) for w in W]; wb=W[int(np.argmax(sc))]
        res.append((seed,wb,score(wb,fit,ev),score((.87,.08,.05),fit,ev),score((1,0,0),fit,ev)))
r=pd.DataFrame(res,columns=['seed','w','tuned','old','nb_only']); d=r.tuned-r.old
print(r.to_string()); print(f'tuned {r.tuned.mean():.5f} old .87/.08/.05 {r.old.mean():.5f} NB only {r.nb_only.mean():.5f} | tuned-old {d.mean():+.5f} +- {d.std()/np.sqrt(len(d)):.5f}, wins {(d>0).sum()}/{len(d)}')
full=[score(w,np.arange(N),np.arange(N)) for w in W]; print('full-OOF best', W[int(np.argmax(full))], round(max(full),5))

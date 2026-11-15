# Nested comparison on the T2 3-model blend OOF: argmax vs fixed-point affine rule (used in sub_v18a) vs Nelder-Mead class weights (report Tech 1).
import numpy as np,pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
y=pd.read_csv('cpe342-kaggle-main/02_multiclass_classification/train.csv').segment.values; S='t2/research/blind_sonnet/'
Bl=.87*np.load('t2/research/best_oof_proba.npy')+.08*np.load(S+'logreg_spline_A_oof.npy')+.05*np.load(S+'mlp_torch_A_oof.npy')
f=lambda p,yy:f1_score(yy,p,average='macro')
def affine(Pa,ya):
    pi=np.bincount(ya,minlength=4)/len(ya);q=pi.copy();F=np.full(4,.75)
    for _ in range(8):
        a=2/(pi+q);b=-F/(pi+q);pr=(a*Pa+b).argmax(1);q=np.bincount(pr,minlength=4)/len(ya);F=np.array([2*((pr==k)&(ya==k)).mean()/(pi[k]+q[k]) for k in range(4)])
    return lambda P:(a*P+b).argmax(1)
def nm(Pa,ya):
    obj=lambda v:-f((Pa*np.r_[1,np.exp(v)]).argmax(1),ya)
    best=min((minimize(obj,x0,method='Nelder-Mead',options=dict(xatol=1e-4,fatol=1e-6,maxiter=400)) for x0 in (np.zeros(3),np.full(3,.05),np.full(3,.1))),key=lambda r:r.fun)
    w=np.r_[1,np.exp(best.x)]; return lambda P:(P*w).argmax(1)
R={'affine':[],'nm':[],'affine_then_nm':[]}
for s in (1,2,3):
    for a,b in StratifiedKFold(5,shuffle=True,random_state=s).split(Bl,y):
        base=f(Bl[b].argmax(1),y[b])
        R['affine'].append(f(affine(Bl[a],y[a])(Bl[b]),y[b])-base); R['nm'].append(f(nm(Bl[a],y[a])(Bl[b]),y[b])-base)
for k,v in R.items():
    if v: print(k,'gain vs argmax',round(np.mean(v),5),'+- se',round(np.std(v)/np.sqrt(len(v)),5),'pos',sum(x>0 for x in v),'/',len(v))
d=np.array(R['nm'])-np.array(R['affine']); print('nm - affine',round(d.mean(),5),'+- se',round(d.std()/np.sqrt(len(d)),5))

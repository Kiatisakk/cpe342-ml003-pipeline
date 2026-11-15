# Nested comparison: current rank blend (0.7/0.2/0.1) vs non-negative logistic stack of logits (report Tech 7), binary F2 with inner cut.
import numpy as np,pandas as pd
from scipy.stats import rankdata
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
l=pd.read_csv('cpe342-kaggle-main/01_binary_classification/train.csv'); m=l.is_cheater.notna().values; y=l[m].is_cheater.astype(int).values
lg=lambda p:np.log(np.clip(p,1e-5,1-1e-5)/(1-np.clip(p,1e-5,1-1e-5)))
M=[np.load('t1/research/best_oof_avg3.npy'),np.load('t1/research/blind_sonnet/stack_lr_oof.npy')[m],np.load('t1/preds/t1_pipeline/oof_blend.npy')]
Z=np.column_stack([lg(v) for v in M]); R=np.column_stack([rankdata(v)/len(v) for v in M])
def F2(p,yy): tp=(p&(yy==1)).sum();fp=(p&(yy==0)).sum();fn=(~p&(yy==1)).sum();return 5*tp/(5*tp+4*fn+fp)
def cut(s,yy): q=np.quantile(s,np.linspace(.30,.50,81)); return q[np.argmax([F2(s>=x,yy) for x in q])]
def nnls_logit(Za,ya,l2=1e-3):
    def obj(v):
        w=np.exp(v[1:]); z=v[0]+Za@w; p=1/(1+np.exp(-z)); return -np.mean(ya*np.log(p+1e-12)+(1-ya)*np.log(1-p+1e-12))+l2*np.sum(w**2)
    v=minimize(obj,np.r_[0,np.log(np.full(Za.shape[1],.3))],method='L-BFGS-B').x; return v[0],np.exp(v[1:])
g={'rank721':[],'nn_stack':[]}; W=[]
for s in (1,2,3):
    for a,b in StratifiedKFold(5,shuffle=True,random_state=s).split(Z,y):
        base=F2(M[0][b]>=cut(M[0][a],y[a]),y[b])
        r=R@np.array([.7,.2,.1]); g['rank721'].append(F2(r[b]>=cut(r[a],y[a]),y[b])-base)
        c,w=nnls_logit(Z[a],y[a]); W.append(w); z=c+Z@w; g['nn_stack'].append(F2(z[b]>=cut(z[a],y[a]),y[b])-base)
for k,v in g.items(): print(k,'gain vs ours',round(np.mean(v),5),'+- se',round(np.std(v)/np.sqrt(15),5),'pos',sum(x>0 for x in v),'/15')
print('stack weights mean',np.mean(W,0).round(3)); d=np.array(g['nn_stack'])-np.array(g['rank721']); print('stack - rank721',round(d.mean(),5),'+- se',round(d.std()/np.sqrt(15),5))

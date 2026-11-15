import pandas as pd,numpy as np,warnings;warnings.filterwarnings('ignore')
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from scipy.ndimage import gaussian_filter1d
t=pd.read_csv('cpe342-kaggle-main/02_multiclass_classification/train.csv');y=t.segment.astype(int).values;K=4
NOISE=['random_metric_1','random_metric_2','random_metric_3','account_age_days','responds_to_discounts','item_type_preference_social','avg_match_length','preferred_game_mode','region','platform','device_type','payment_method','language','account_status','player_type_tag','skill_tier']
FEATS=[c for c in t.columns if c not in['player_id','segment']+NOISE]
DISC=[c for c in FEATS if (not pd.api.types.is_numeric_dtype(t[c])) or t[c].nunique()<=12]
CONT=[c for c in FEATS if c not in DISC]
folds=list(StratifiedKFold(5,shuffle=True,random_state=42).split(t,y))
f1=lambda L,idx=None:f1_score(y if idx is None else y[idx],L.argmax(1),average='macro')
def loglik(c,a,b,nbins=32,sm=0.0,alpha=1.0,use_nan=False):
    """per-class log p(x|k) for rows b, densities from rows a. returns (len(b),K)"""
    v=t[c]
    if c in DISC:
        s=v.astype(object).where(v.notna(),'NA').astype(str).values;lv=sorted(set(s));m={x:i for i,x in enumerate(lv)};bi=np.array([m[x] for x in s]);n=len(lv)
        nan=(s=='NA')
    else:
        x=v.values.astype(float);e=np.unique(np.nanquantile(x[a],np.linspace(0,1,nbins+1)));n=len(e)-1
        nan=np.isnan(x);bi=np.where(nan,0,np.clip(np.searchsorted(e,x,side='right')-1,0,n-1))
        if use_nan: bi=np.where(nan,n,bi);n+=1
    out=np.zeros((len(b),K))
    for k in range(K):
        sel=a[y[a]==k]
        keep=sel if (use_nan or c in DISC) else sel[~nan[sel]]
        cnt=np.bincount(bi[keep],minlength=n).astype(float)
        if sm>0 and c in CONT: core=gaussian_filter1d(cnt[:n-(1 if use_nan else 0)],sm,mode='nearest');cnt[:len(core)]=core
        lp=np.log((cnt+alpha)/(cnt+alpha).sum());out[:,k]=lp[bi[b]]
    if not (use_nan or c in DISC): out[nan[b]]=0
    return out
def nb_scores(a,b,cfg):
    prior=np.log(np.bincount(y[a],minlength=K)/len(a))
    return prior+sum(loglik(c,a,b,**cfg.get(c,cfg['_default'])) for c in FEATS)

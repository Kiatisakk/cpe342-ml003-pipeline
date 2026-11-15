import pandas as pd,numpy as np,time,warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
D='data/';O='out/'
tr=pd.read_csv(D+'train.csv');te=pd.read_csv(D+'test.csv')
y=tr.segment.values
for d in (tr,te): d['pid']=d.player_id.str[1:].astype(float)
CATS=['region','platform','device_type','payment_method','language','account_status','player_type_tag','engagement_level','loyalty_tier','skill_tier']
NOISE=['random_metric_1','random_metric_2','random_metric_3','preferred_game_mode','account_age_days','avg_match_length','responds_to_discounts','item_type_preference_social']
NUM=[c for c in tr.columns if c not in CATS+['player_id','segment','pid']]
INFO_NUM=[c for c in NUM if c not in NOISE]
INFO_CAT=['engagement_level','loyalty_tier']
skf=StratifiedKFold(5,shuffle=True,random_state=42)
FOLDS=list(skf.split(tr,y))
def mf1(y,p): return f1_score(y,p,average='macro')
def cv(fit_pred,name,Xtr,Xte=None,save=False,log=True):
    """fit_pred(Xa,ya,Xb_list)->list of proba arrays"""
    t=time.time();oof=np.zeros((len(y),4));sc=[];tp=np.zeros((len(Xte),4)) if Xte is not None else None
    for a,b in FOLDS:
        outs=fit_pred(Xtr[a] if not hasattr(Xtr,'iloc') else Xtr.iloc[a],y[a],[Xtr[b] if not hasattr(Xtr,'iloc') else Xtr.iloc[b]]+([Xte] if Xte is not None else []))
        oof[b]=outs[0]; sc.append(mf1(y[b],outs[0].argmax(1)))
        if Xte is not None: tp+=outs[1]/5
    r=(name,np.mean(sc),np.std(sc),time.time()-t)
    print('RESULT %s %.4f +- %.4f  %.0fs'%r,flush=True)
    open(O+'results.tsv','a').write('%s\t%.4f\t%.4f\t%.0f\n'%r)
    if save: np.save(O+name+'_oof.npy',oof);np.save(O+name+'_test.npy',tp)
    return oof,tp
def prep_gbm(withid):
    cols=NUM+(['pid'] if withid else [])
    A=tr[cols].copy();B=te[cols].copy()
    for c in CATS:
        cat=pd.Categorical(tr[c].fillna('NA'));A[c]=cat.codes;B[c]=pd.Categorical(te[c].fillna('NA'),categories=cat.categories).codes
    return A,B

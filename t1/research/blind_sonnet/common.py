import pandas as pd,numpy as np,time,json,os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import fbeta_score,roc_auc_score
D='data/';O='out/'
tr=pd.read_csv(D+'train.csv');te=pd.read_csv(D+'test.csv')
SIG=['reports_received','crosshair_placement','kill_death_ratio','headshot_percentage','game_sense_score','accuracy_score','level','spray_control_score','aiming_smoothness','win_rate','account_age_days','friend_network_size','kill_consistency','reaction_time_ms','level_progression_speed']
ALL=[c for c in tr.columns if c not in('player_id','is_cheater')]
lab=tr.is_cheater.notna().values; y=tr.is_cheater.values
def feats(df,cols=None,eng=True):
    X=df[cols or ALL].copy()
    if eng:
        a,l,s=df.account_age_days,df.level,df.level_progression_speed
        X['age_lo']=(l*.15/s).where(a.isna(),a);X['age_hi']=(l*1.08/s).where(a.isna(),a)
        X['lv_lo']=(s*a/1.08).where(l.isna(),l);X['lv_hi']=(s*a/.15).where(l.isna(),l)
        X['r']=s*a/l
        X['fr_age']=df.friend_network_size/a
        X['rep_age']=df.reports_received/a
    return X
def f2_thr(p,yy):
    best=(-1,.5)
    for t in np.quantile(p,np.linspace(.2,.8,61)):
        f=fbeta_score(yy,p>=t,beta=2,zero_division=0)
        if f>best[0]:best=(f,t)
    return best[1]
def run(name,fit_predict,Xtr,Xte,inner=3,save=False,final_test=True,sub=None,verbose=True):
    """fit_predict(Xa,ya,Xb_list)->list of prob arrays. Nested threshold via inner CV on train fold."""
    t0=time.time();idx=np.where(lab)[0];yl=y[idx].astype(int)
    oof=np.full(len(tr),np.nan);scores=[];aucs=[]
    for k,(a,b) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(idx,yl)):
        ia,ib=idx[a],idx[b]
        pb,=fit_predict(Xtr[ia],y[ia].astype(int),[Xtr[ib]])
        ip=np.zeros(len(ia))
        for a2,b2 in StratifiedKFold(inner,shuffle=True,random_state=k).split(ia,y[ia]):
            ip[b2]=fit_predict(Xtr[ia[a2]],y[ia[a2]].astype(int),[Xtr[ia[b2]]])[0]
        thr=f2_thr(ip,y[ia].astype(int))
        oof[ib]=pb;scores.append(fbeta_score(y[ib].astype(int),pb>=thr,beta=2));aucs.append(roc_auc_score(y[ib],pb))
    r=dict(name=name,f2=float(np.mean(scores)),std=float(np.std(scores)),auc=float(np.mean(aucs)),sec=time.time()-t0)
    if save:
        np.save(O+name+'_oof.npy',oof)
        pt,=fit_predict(Xtr[idx],yl,[Xte]);np.save(O+name+'_test.npy',pt)
    with open(O+'results.jsonl','a') as f:f.write(json.dumps(r)+'\n')
    if verbose:print(r,flush=True)
    return oof,r

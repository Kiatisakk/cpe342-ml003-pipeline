import warnings;warnings.filterwarnings('ignore')
import numpy as np,pandas as pd
from feat import metrics,hav
STAT=['account_age_days','behavior_stability','password_changed','vpn_usage','suspicious_login_time','mass_item_sale']
def build(df):
    cols={}
    for m in metrics:
        if m in('login_lat','login_lon'):continue
        X=df[[f'{m}_{i}' for i in range(1,5)]].values
        b=np.nanmean(X[:,:3],1); b12=np.nanmean(X[:,:2],1)
        cols[m+'_b']=b
        cols[m+'_sd']=np.nanstd(X[:,:3],1)
        cols[m+'_d4']=X[:,3]-b
        cols[m+'_d3']=X[:,2]-b12
    L=lambda k,i:df[f'login_{k}_{i}'].values
    hl=np.nanmean([L('lat',1),L('lat',2)],0);ho=np.nanmean([L('lon',1),L('lon',2)],0)
    for p in(3,4):
        cols[f'geo{p}']=hav(hl,ho,L('lat',p),L('lon',p))
        cols[f'latbad{p}']=(np.abs(L('lat',p))>90).astype(float)
    for c in STAT: cols[c]=df[c].values
    F=pd.DataFrame(cols)
    return F
def prep(F):
    A=F.values.astype(float)
    med=np.nanmedian(A,0); mad=1.4826*np.nanmedian(np.abs(A-med),0)
    sd=np.nanstd(A,0); s=np.where(mad>1e-9,mad,np.where(sd>1e-9,sd,1))
    Z=(A-med)/s
    Z=np.where(np.isnan(Z),0,Z)
    return np.clip(Z,-25,25)
def shuffle_cols(df,rng):
    o=df.copy()
    for c in o.columns:
        if c=='player_id':continue
        o[c]=o[c].values[rng.permutation(len(o))]
    return o
def shuffle_blocks(df,rng):
    o=df.copy(); n=len(o)
    groups=[]
    for m in metrics:
        if m in('login_lat','login_lon'):continue
        groups.append([f'{m}_{i}' for i in range(1,5)])
    groups.append([f'login_{k}_{i}' for i in range(1,5) for k in('lat','lon')])
    groups+= [[c] for c in STAT]
    for g in groups:
        o[g]=df[g].values[rng.permutation(n)]
    return o

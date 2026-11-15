import warnings;warnings.filterwarnings('ignore')
import numpy as np,pandas as pd,json
from feat import hav
from fe import shuffle_cols,shuffle_blocks
def rules(df):
    X=lambda m,i:df[f'{m}_{i}'].values
    b123=lambda m:np.nanmean(np.c_[X(m,1),X(m,2),X(m,3)],1)
    def lr(m):
        s=0.05*np.nanstd(X(m,4)); return np.nan_to_num(np.log((X(m,4)+s)/(b123(m)+s)))
    lat=lambda i:X('login_lat',i); lon=lambda i:X('login_lon',i)
    hl=np.nanmean([lat(1),lat(2)],0);ho=np.nanmean([lon(1),lon(2)],0)
    j3=np.nan_to_num(hav(hl,ho,lat(3),lon(3)));j4=np.nan_to_num(hav(hl,ho,lat(4),lon(4)))
    g3=(j3>5)|(np.abs(np.nan_to_num(lat(3)))>90); g4=(j4>5)|(np.abs(np.nan_to_num(lat(4)))>90)
    m3=lambda m:np.nanmean(np.c_[X(m,1),X(m,2),X(m,3)],1)
    sk=lr('skill_rating')>0.26; st=np.nan_to_num(X('skill_trajectory',4)-b123('skill_trajectory'))>25
    C=sk|st
    pf=lr('purchase_frequency')>0.9; pam=lr('purchase_amount_mean')>0.55; play=lr('play_pattern_consistency')<-0.3
    A=g3|(pf&pam)|(g4&(pf|pam)&~play)
    B=(play|g4)&~A&~C
    f=lambda c,t:np.nan_to_num(df[c].values,nan=9e9)<t
    bot=(np.nan_to_num(m3('ip_hash_entropy'),nan=9)<0.25)|(np.nan_to_num(m3('login_time_variance'),nan=9)<1.0)|(np.nan_to_num(m3('session_duration_std'),nan=99)<6.5)|(np.nan_to_num(m3('kda_std'),nan=9)<0.146)
    comm=np.nan_to_num(m3('communication_frequency'),nan=9)<0.17
    act=np.nan_to_num(m3('activity_level'),nan=9)<0.34
    beh=np.nan_to_num(df.behavior_stability.values,nan=9)<0.55
    G={'takeover_A_vpn_geo3_purchase':A,'takeover_B_pwchange_geo4_play':B,'perf_spike_C':C,'bot_lowvariance':bot,'low_comm':comm,'low_activity':act,'low_behavior_stability':beh}
    # priority resolve overlaps
    return G
if __name__=='__main__':
    df=pd.read_csv('data/test.csv'); G=rules(df)
    for k,v in G.items(): print(k,v.sum())
    M=np.c_[list(G.values())].T; print('overlap',(M.sum(1)>1).sum(),'union',M.any(1).sum())
    res={'real':{k:int(v.sum()) for k,v in G.items()}}
    for kind,fn in (('colshuffle',shuffle_cols),('blockshuffle',shuffle_blocks)):
        cs=[]
        for sd in range(5):
            cs.append({k:int(v.sum()) for k,v in rules(fn(df,np.random.RandomState(sd))).items()})
        res[kind]=cs; print(kind,pd.DataFrame(cs).agg(['mean','max']).round(1).to_dict())
    # half-subsample consistency: rule counts on random halves (x2)
    hs=[]
    for sd in range(5):
        idx=np.random.RandomState(100+sd).choice(len(df),len(df)//2,replace=False)
        hs.append({k:int(v.sum())*2 for k,v in rules(df.iloc[idx].reset_index(drop=True)).items()})
    res['half_x2']=hs; print('halfx2',pd.DataFrame(hs).agg(['mean','min','max']).round(0).to_dict())
    json.dump(res,open('out/rule_null.json','w'),indent=1)
    # output files (priority order for overlaps)
    names=list(G); lab=np.array(['']*len(df),dtype=object)
    for k in reversed(names): lab[G[k]]=k
    ix=np.where(lab!='')[0]
    pd.DataFrame({'row_index':ix,'group_name':lab[ix]}).to_csv('out/blind_groups.csv',index=False)
    print('written',len(ix))

import warnings;warnings.filterwarnings('ignore')
import numpy as np,pandas as pd,json
from feat import hav
from fe import shuffle_blocks
def co(df):
    X=lambda m,i:df[f'{m}_{i}'].values
    b=lambda m:np.nanmean(np.c_[X(m,1),X(m,2),X(m,3)],1)
    def lr(m):
        s=0.05*np.nanstd(X(m,4)); return np.nan_to_num(np.log((X(m,4)+s)/(b(m)+s)))
    lat=lambda i:X('login_lat',i); lon=lambda i:X('login_lon',i)
    hl=np.nanmean([lat(1),lat(2)],0);ho=np.nanmean([lon(1),lon(2)],0)
    j3=np.nan_to_num(hav(hl,ho,lat(3),lon(3)));j4=np.nan_to_num(hav(hl,ho,lat(4),lon(4)))
    g3=j3>5;g4=j4>5
    sk=lr('skill_rating')>0.26; st=np.nan_to_num(X('skill_trajectory',4)-b('skill_trajectory'))>25
    pf=lr('purchase_frequency')>0.9; pam=lr('purchase_amount_mean')>0.55; play=lr('play_pattern_consistency')<-0.3
    vpn=df.vpn_usage.values==1;pw=df.password_changed.values==1
    return {'A: g3&(pf|pam)':int((g3&(pf|pam)).sum()),'A: g3&vpn':int((g3&vpn).sum()),'A: g3&(pf|pam)&vpn':int((g3&(pf|pam)&vpn).sum()),
            'B: play&g4':int((play&g4).sum()),'B: play&pw':int((play&pw).sum()),'B: play&g4&pw':int((play&g4&pw).sum()),
            'C: sk&st':int((sk&st).sum()),'C: sk&acc':int((sk&(lr('accuracy_mean')>.2)).sum())}
df=pd.read_csv('data/test.csv');r=co(df);print('real',r)
n=pd.DataFrame([co(shuffle_blocks(df,np.random.RandomState(s))) for s in range(5)]);print(n.agg(['mean','max']).round(1).T)
json.dump({'real':r,'block_null':n.to_dict('list')},open('out/cooc.json','w'))

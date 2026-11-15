import pandas as pd, numpy as np
from sklearn.metrics import f1_score
R='cpe342-kaggle-main/02_multiclass_classification/'
tr=pd.read_csv(R+'train.csv'); tr['id']=tr.player_id.str[1:].astype(int)
y=tr.segment.astype(int).values; blk=np.digitize(tr.id,[52000,84500,110500]); mm=y!=blk
P=np.load('t2/research/best_oof_proba.npy'); print(P.shape, P.argmax(1).__eq__(y).mean())
print('f1 nb',f1_score(y,P.argmax(1),average='macro'),'block',f1_score(y,blk,average='macro'))
# NB likelihood of block label vs stated label on mismatches
pb=P[np.arange(len(y)),blk]; ps=P[np.arange(len(y)),y]
print('mismatch: mean P(block)',pb[mm].mean(),'mean P(stated)',ps[mm].mean(),' argmax==block',(P.argmax(1)[mm]==blk[mm]).mean(),'==stated',(P.argmax(1)[mm]==y[mm]).mean())
print('match: argmax==label',(P.argmax(1)[~mm]==y[~mm]).mean())
# expected if features follow block: NB acc on matched ~0.77
# if features follow stated: argmax==stated would be ~.77
# combined posterior with prior from block: P(true=k)= noise model
eps=0.03
def comb(P,blk,eps):
    pr=np.full(P.shape,eps/3); pr[np.arange(len(P)),blk]=1-eps
    Q=P*pr; return Q/Q.sum(1,keepdims=True)
Q=comb(P,blk,.03)
print('comb acc',(Q.argmax(1)==y).mean(),(Q.argmax(1)==blk).mean())
# NB confidence on mismatch rows when NB strongly disagrees
conf=P.max(1)
for t in [.9,.95,.99,.999]:
    s=(conf>t); print(t,s.sum(),'agree w/ block among confident',(P.argmax(1)[s]==blk[s]).mean(),'mismatch share',mm[s].mean(),'NB-conf-nonblock count',(s&(P.argmax(1)!=blk)).sum(),'of which label==nbpred',(s&(P.argmax(1)!=blk)&(P.argmax(1)==y)).sum())
# missingness / column association with mm
num=tr.select_dtypes('number').drop(columns=['segment','id'])
print('NaN rate diff',(num.isna().mean(axis=1)[mm].mean(),num.isna().mean(axis=1)[~mm].mean()))
print(tr.id[mm].mod(2).mean(), tr.id[mm].mod(10).value_counts().sort_index().tolist())
for c in ['region','platform','device_type','payment_method','language','account_status','player_type_tag','skill_tier','loyalty_tier','engagement_level']:
    if c in tr:
        ct=pd.crosstab(tr[c].fillna('NA'),mm,normalize='index')[True]; print(c,ct.round(3).to_dict())

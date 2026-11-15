import pandas as pd, numpy as np
from sklearn.metrics import f1_score
R='cpe342-kaggle-main/02_multiclass_classification/'
tr=pd.read_csv(R+'train.csv'); y=tr.segment.astype(int).values
blk=np.digitize(tr.player_id.str[1:].astype(int),[52000,84500,110500])
P=np.load('t2/research/best_oof_proba.npy')
base=f1_score(y,blk,average='macro'); print('base',base)
pb=P[np.arange(len(y)),blk]
for T in [5,10,20,40,80,160]:
    Pm=P.copy(); Pm[np.arange(len(y)),blk]=0; o=Pm.argmax(1); ov=Pm.max(1)/pb>T
    pred=np.where(ov,o,blk); print(T,ov.sum(),f1_score(y,pred,average='macro'),(pred==y).mean())
# per-class: f1 by class
print(f1_score(y,blk,average=None))
# 1000 random-flip simulations: expected F1 sd on 25889 rows
rng=np.random.default_rng(0); s=[]
for _ in range(200):
    idx=rng.choice(len(y),25889); s.append(f1_score(y[idx],blk[idx],average='macro'))
print('sub-sample F1 mean/sd',np.mean(s),np.std(s))

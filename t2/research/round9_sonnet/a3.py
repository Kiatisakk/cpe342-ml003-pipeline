import pandas as pd, numpy as np
from sklearn.metrics import f1_score
R='cpe342-kaggle-main/02_multiclass_classification/'
tr=pd.read_csv(R+'train.csv'); te=pd.read_csv(R+'test.csv')
ID=tr.player_id.str[1:].astype(int).values; IT=te.player_id.str[1:].astype(int).values
y=tr.segment.astype(int).values
o=np.argsort(ID); ID,y=ID[o],y[o]
for k,c0 in enumerate([52000,84500,110500]):
    gain=np.cumsum((y==k).astype(int)-(y==k+1).astype(int))
    # all positions achieving max within window
    best=gain.max(); pos=np.where(gain==best)[0]
    print(k,'best',best,'cuts after ids',ID[pos][:5],ID[pos+1][:5],'| gain at exact cut',gain[np.searchsorted(ID,c0)-1])
    for d in [-50,-10,-1,0,1,10,50]:
        print('   cut',c0+d,'gain',gain[np.searchsorted(ID,c0+d)-1],end=';')
    print()
    # local noise rate near cut: labels within +-200
    w=(ID>=c0-200)&(ID<c0+200); print('  local mismatch',((y[w]!=np.digitize(ID[w],[52000,84500,110500]))).mean(), 'test ids within +-20:',IT[(IT>=c0-20)&(IT<c0+20)])
# is distance to boundary related to mismatch? (any bin)
blk=np.digitize(ID,[52000,84500,110500]); mm=y!=blk
d=np.min(np.abs(ID[:,None]-np.array([52000,84500,110500])[None]),1)
for lo,hi in [(0,50),(50,500),(500,5000),(5000,1e9)]:
    s=(d>=lo)&(d<hi); print(lo,hi,s.sum(),mm[s].mean())
# mismatch destination structure: adjacent vs far
print(pd.crosstab(blk[mm],y[mm]))
# run structure of mismatches: autocorr of mm in id order
print('P(mm next|mm)',mm[1:][mm[:-1]].mean(),'base',mm.mean())

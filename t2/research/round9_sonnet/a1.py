import pandas as pd, numpy as np
R='cpe342-kaggle-main/02_multiclass_classification/'
tr=pd.read_csv(R+'train.csv'); te=pd.read_csv(R+'test.csv')
tr['id']=tr.player_id.str[1:].astype(int); te['id']=te.player_id.str[1:].astype(int)
print(tr.id.min(),tr.id.max(),te.id.min(),te.id.max(),len(tr),len(te))
print(tr.segment.value_counts(dropna=False))
c=[52000,84500,110500]
tr['blk']=np.digitize(tr.id,c)
l=tr.dropna(subset=['segment']).copy(); l['segment']=l.segment.astype(int)
m=l[l.segment!=l.blk]; print(len(l),len(m),len(m)/len(l))
print(pd.crosstab(l.blk,l.segment))
print(l.groupby('blk').apply(lambda d:(d.segment!=d.blk).mean()))
# boundary proximity
for cc in c:
    d=l[(l.id>=cc-300)&(l.id<cc+300)].sort_values('id'); print(cc,d[['id','segment','blk']].assign(x=d.segment!=d.blk).query('x').id.tolist())
# parity/mod
for k in [2,3,4,5,7,10,100]:
    print(k,(m.id%k).value_counts(normalize=True).round(3).sort_index().tolist()[:10])
# gaps
print(m.id.diff().describe())

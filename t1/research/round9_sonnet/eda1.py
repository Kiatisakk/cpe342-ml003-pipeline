import pandas as pd,numpy as np
D='cpe342-kaggle-main/01_binary_classification/'
a=pd.read_csv(D+'train.csv');b=pd.read_csv(D+'test.csv')
a['n']=a.player_id.str[1:].astype(int);b['n']=b.player_id.str[1:].astype(int)
print(a.n.min(),a.n.max(),a.n.nunique(),len(a),b.n.min(),b.n.max(),b.n.nunique())
print('dup ids train',a.player_id.duplicated().sum(),'overlap tr/te',len(set(a.player_id)&set(b.player_id)))
l=a[a.is_cheater.notna()]
for m in [2,3,5,7,10,16,100,1000]:
    g=l.groupby(l.n%m).is_cheater.mean();print('mod',m,round(g.min(),3),round(g.max(),3))
for w in [10,50,100,500]:
    g=l.groupby(l.n//w).is_cheater.agg(['mean','size']);print('blk',w,g['mean'].std(),np.sqrt(.35*.65/g['size'].mean()))
# row order
l=l.reset_index(drop=True)
print('order blocks',l.groupby(l.index//1000).is_cheater.mean().std(), 'autocorr',l.is_cheater.autocorr())
print('unlab frac by id',a.groupby(a.n//10000).is_cheater.apply(lambda s:s.isna().mean()).round(3).tolist())
fe=[c for c in a.columns if c not in('player_id','n','is_cheater')]
# row-duplicates on features
x=pd.concat([a[fe],b[fe]]);print('dup feature rows',x.duplicated().sum())
# missingness by class
print((l[fe].isna().groupby(l.is_cheater).mean().T).assign(d=lambda d:d[1.0]-d[0.0]).round(3).to_string())
# decimals
def dec(s):
    s=s.dropna().astype(str);return s.str.split('.').str[1].str.len().fillna(0)
for c in fe:
    d=dec(l[c]);
    print(c,'maxdec',d.max(),'mean dec by class',[round(dec(l.loc[l.is_cheater==k,c]).mean(),2) for k in (0,1)],'nuniq',l[c].nunique())

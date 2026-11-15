import pandas as pd, numpy as np, hashlib, os
from PIL import Image
R='cpe342-kaggle-main/04_image_classification/'
tr=pd.read_csv(R+'train.csv'); te=pd.read_csv(R+'test.csv')
def info(d,i):
    p=f'{R}{d}/{i}.jpg'; b=open(p,'rb').read()
    im=Image.open(p); 
    q=getattr(im,'quantization',{}); 
    return hashlib.md5(b).hexdigest(), len(b), im.size, (sum(q[0][:8]) if q else -1)
rng=np.random.default_rng(0)
trs=tr.sample(6000,random_state=0); tes=te.sample(6000,random_state=0)
A=[(i,c)+info('train',i) for i,c in zip(trs.id,trs['class'])]
B=[(i,)+info('test',i) for i in tes.id]
a=pd.DataFrame(A,columns=['id','cls','md5','size','dim','q']); b=pd.DataFrame(B,columns=['id','md5','size','dim','q'])
print(a.dim.value_counts().head(), b.dim.value_counts().head(), a.q.value_counts().head(3).to_dict(), b.q.value_counts().head(3).to_dict())
print('exact md5 overlap sample:',len(set(a.md5)&set(b.md5)),' dup within train', a.md5.duplicated().sum(),' within test',b.md5.duplicated().sum())
print(a.groupby('cls')['size'].describe())
# id features
tr2=tr.copy(); tr2['len']=tr2.id.str.len(); tr2['first2']=tr2.id.str[:2]; tr2['c1']=tr2.id.str[1]
print(pd.crosstab(tr2.c1,tr2['class'],normalize='columns').round(3).iloc[:12])
print('id len',tr2.len.value_counts().to_dict())
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import HistGradientBoostingClassifier as H
X=np.array([[ord(ch) for ch in s.ljust(7)[:7]] for s in tr.id]); 
print('id-char CV acc',cross_val_score(H(max_iter=100),X,tr['class'],cv=3).mean(),'majority',tr['class'].value_counts(normalize=True).max())
# sorted-id position vs class
s=tr.sort_values('id').reset_index(drop=True); 
print('adjacent same-class rate (sorted by id):',(s['class'].values[1:]==s['class'].values[:-1]).mean(),' expected',(tr['class'].value_counts(normalize=True)**2).sum())
print('adjacent same-class (csv order):',(tr['class'].values[1:]==tr['class'].values[:-1]).mean())
# test id sortedness
print('test.csv sorted by id?',te.id.is_monotonic_increasing, ' train sorted?',tr.id.is_monotonic_increasing)
lst=sorted(os.listdir(R+'test'))[:3]; print(lst, te.id.head(3).tolist())
print('ids overlap train/test', len(set(tr.id)&set(te.id)))
a['size_cls']=a.groupby('cls')['size'].transform('mean')

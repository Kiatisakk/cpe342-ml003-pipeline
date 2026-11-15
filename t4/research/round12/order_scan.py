# Offline scan: is there a row order, subset or other column whose class mix explains the LB digit shares?
import pandas as pd,numpy as np,itertools,os,glob
SH=np.array([.136,.298,.24,.211,.115]); R='cpe342-kaggle-main/'
T=pd.read_csv(R+'04_image_classification/test.csv').id.values
s=pd.read_csv('t4/research/t4_test_str.csv'); C=sorted(s.pred.unique()); y=s.pred.map({c:i for i,c in enumerate(C)}).values
tr=pd.read_csv(R+'04_image_classification/train.csv'); print('classes',C)
def best(mix):
    return min((np.abs(mix[list(p)]-SH).max(),p) for p in itertools.permutations(range(5)))
print('test mix',np.bincount(y,minlength=5)/len(y),best(np.bincount(y,minlength=5)/len(y)))
trm=tr['class'].value_counts(normalize=True).reindex(C).values; print('train mix',trm,best(trm))
# 1 contiguous windows, in test.csv order and sorted-id order
for nm,o in [('csv',np.arange(len(y))),('sorted',np.argsort(T))]:
    yy=y[o]; res=[]
    for L in [2000,4000,7767,12945,18122]:
        for a in range(0,len(y)-L+1,250):
            m=np.bincount(yy[a:a+L],minlength=5)/L; res.append((*best(m),L,a))
    res.sort(key=lambda r:r[0]); print(nm,'best windows (maxdiff,perm,L,start)',res[:3])
    w=[np.bincount(yy[a:a+2000],minlength=5)/2000 for a in range(0,len(y)-2000,2000)]
    print(nm,'window mix sd',np.std(w,0).round(4),'iid sd',np.sqrt(np.bincount(y)/len(y)*(1-np.bincount(y)/len(y))/2000).round(4))
# 2 discrete columns of other tasks' test files
rows=[]
for f in glob.glob(R+'0[1235]_*/test.csv'):
    d=pd.read_csv(f)
    for c in d.columns:
        v=d[c].dropna()
        if v.nunique()==5:
            m=v.value_counts(normalize=True).values
            rows.append((np.abs(np.sort(m)-np.sort(SH)).max(),os.path.basename(os.path.dirname(f)),c,m.round(3).tolist()))
print('5-level columns'); [print(r) for r in sorted(rows,key=lambda r:r[0])[:8]]
# 3 file metadata
ft=[os.stat(R+'04_image_classification/test/'+i+'.jpg') for i in T]
mt=np.array([x.st_mtime for x in ft]); sz=np.array([x.st_size for x in ft])
print('mtime unique',len(np.unique(mt)),'range s',mt.max()-mt.min())
print('corr rowindex~size',np.corrcoef(np.arange(len(T)),sz)[0,1])

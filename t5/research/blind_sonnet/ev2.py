import warnings;warnings.filterwarnings('ignore')
from feat import *
pd.set_option('display.width',250,'display.max_rows',500)
Z4=pd.read_pickle('out/Z4.pkl');g3=np.load('out/g3.npy');g4=np.load('out/g4.npy');F=pd.read_pickle('out/stat_flags.pkl')
rest=~(g3|g4|F.any(axis=1).values)
cols=list(Z4.columns)
big=(Z4.abs()>5)&rest[:,None]
sig=np.sign(Z4).fillna(0)*big
for lab,mask in (('rest',rest),('geo4only',g4&~g3),('geo3',g3)):
    b=((Z4.abs()>5)&mask[:,None]); s=np.sign(Z4).fillna(0)*b
    n=b.sum(1)
    print(lab,mask.sum(),'rows with >=1 big',int((n>=1).sum()),' >=2',int((n>=2).sum()))
    keys=s[n>=1].apply(lambda r:' '.join(f"{'+' if v>0 else '-'}{c}" for c,v in r.items() if v!=0),axis=1)
    print(keys.value_counts().head(25))

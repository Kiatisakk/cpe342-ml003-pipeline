import warnings;warnings.filterwarnings('ignore')
from feat import *
pd.set_option('display.width',250)
D=pd.read_pickle('out/D4.pkl');g3=np.load('out/g3.npy');g4=np.load('out/g4.npy');F=pd.read_pickle('out/stat_flags.pkl')
def X(m,i):return d[f'{m}_{i}'].values
def base(m):return np.nanmean(np.c_[X(m,1),X(m,2),X(m,3)],1)
C=(D.skill_rating>0.26).values
Ct=(D.skill_trajectory>25).values
B=(D.play_pattern_consistency<-0.3).values
print('C',C.sum(),'Ct',Ct.sum(),'C&Ct',(C&Ct).sum(),'B',B.sum())
loc=np.log((X('location_changes',4)+.01)/(base('location_changes')+.01))
print('loc4 hist',np.histogram(np.nan_to_num(loc),bins=[-3,-.5,0,.5,1,1.5,2,2.5,3,4,6])[0])
cur=X('currency_transfers',4)/ (base('currency_transfers')+.1)
print('cur ratio hist',np.histogram(np.nan_to_num(cur),bins=[0,.5,1,1.5,2,3,4,6,10,30,100,1000])[0])
pf=X('purchase_frequency',4)/(base('purchase_frequency')+.1)
print('pf ratio hist',np.histogram(np.nan_to_num(pf),bins=[0,.5,1,1.5,2,3,4,6,10,30,100,1000])[0])
for nm,mk in (('C',C),('B',B),('g3',g3),('g4',g4)):
    print(nm,mk.sum(),{k:int((mk&v).sum()) for k,v in (('C',C),('B',B),('g3',g3),('g4',g4),('stat',F.any(axis=1).values))})
print('C-only (no g3/g4/B):',(C&~g3&~g4&~B).sum(),' Ct-only not C',(Ct&~C).sum())
print('B & ~g4',(B&~g4).sum(),'g4&~B&~g3',(g4&~B&~g3).sum(), ' g3&~g4',(g3&~g4).sum())

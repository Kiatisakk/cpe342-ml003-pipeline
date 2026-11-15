import warnings;warnings.filterwarnings('ignore')
from feat import *
pd.set_option('display.width',250,'display.max_rows',500)
L=lambda k,i:d[f'login_{k}_{i}'].values
hl=np.nanmean([L('lat',1),L('lat',2)],0);ho=np.nanmean([L('lon',1),L('lon',2)],0)
j3=hav(hl,ho,L('lat',3),L('lon',3));j4=hav(hl,ho,L('lat',4),L('lon',4))
g3=np.nan_to_num(j3)>5; g4=np.nan_to_num(j4)>5
g3|= np.abs(np.nan_to_num(L('lat',3)))>90
g4|= np.abs(np.nan_to_num(L('lat',4)))>90
print('g3',g3.sum(),'g4',g4.sum(),'both',(g3&g4).sum())
Z4=pd.read_pickle('out/Z4.pkl');Z3=pd.read_pickle('out/Z3.pkl')
big4=Z4.abs()>6; big3=Z3.abs()>6
print('P(big Z4 metric | geo4) vs | no geo');
print(pd.DataFrame({'geo4':big4[g4].mean(),'nogeo':big4[~g4&~g3].mean(),'geo3':big4[g3].mean()}).round(3))
print(pd.DataFrame({'geo3':big3[g3].mean(),'geo4':big3[g4].mean(),'nogeo':big3[~g4&~g3].mean()}).round(3))
np.save('out/g3.npy',g3);np.save('out/g4.npy',g4)

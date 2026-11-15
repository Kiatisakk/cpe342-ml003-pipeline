import warnings;warnings.filterwarnings('ignore')
from feat import *
L=lambda k,i:d[f'login_{k}_{i}'].values
# period 1 vs 2 distance
d12=hav(L('lat',1),L('lon',1),L('lat',2),L('lon',2))
print('d12 q',np.nanpercentile(d12,[50,90,99,99.9,100]).round(2))
hl=np.nanmean([L('lat',1),L('lat',2)],0);ho=np.nanmean([L('lon',1),L('lon',2)],0)
for p in (3,4):
    dj=hav(hl,ho,L('lat',p),L('lon',p))
    print(p,'q',np.nanpercentile(dj,[50,90,99,99.5,99.9,100]).round(2))
    h,e=np.histogram(dj[~np.isnan(dj)],bins=[0,1,2,3,4,5,6,8,10,15,20,30,50,100,181]);print(h)
    print(' invalid lat',np.nansum(np.abs(L('lat',p))>90),' |lon|>180',np.nansum(np.abs(L('lon',p))>180))
for k in('lat','lon'):
  for p in(1,2): print(k,p,'out-of-range', np.nansum(np.abs(L(k,p))>(90 if k=='lat' else 180)))

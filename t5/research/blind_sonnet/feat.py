import pandas as pd,numpy as np
d=pd.read_csv('data/test.csv')
metrics=sorted({c[:-2] for c in d.columns if c[-2:] in('_1','_2','_3','_4')})
def mad(v): 
    v=v[~np.isnan(v)]; return 1.4826*np.median(np.abs(v-np.median(v)))+1e-9
def hav(la1,lo1,la2,lo2):
    la1,lo1,la2,lo2=map(np.radians,(la1,lo1,la2,lo2))
    a=np.sin((la2-la1)/2)**2+np.cos(la1)*np.cos(la2)*np.sin((lo2-lo1)/2)**2
    return np.degrees(2*np.arcsin(np.sqrt(np.clip(a,0,1))))

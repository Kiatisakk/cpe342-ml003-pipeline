import numpy as np,pandas as pd,time
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from fe import *
df=pd.read_csv('data/test.csv');F=build(df);Z=prep(F)
M=np.load('out/rule_M.npy');names=list(pd.read_csv('out/rule_names.csv').iloc[:,0])
lab=np.where(M.any(1),M.argmax(1),-1)
# features emphasising structure: robustly clipped (+-8), PCA 15
Zc=np.clip(Z,-8,8); P=PCA(15,random_state=0).fit_transform(Zc)
t=time.time(); h=HDBSCAN(min_cluster_size=80,min_samples=15).fit(P); print('hdbscan',round(time.time()-t),'s')
ct=pd.crosstab(h.labels_,pd.Series(lab).map(lambda i:'normal' if i<0 else names[i][:12]))
print(ct)
np.save('out/hdbscan_label.npy',h.labels_)
np.save('out/hdbscan_score.npy',1-h.probabilities_ + (h.labels_<0)) 

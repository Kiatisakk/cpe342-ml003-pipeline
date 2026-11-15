import numpy as np,pandas as pd,sys,time
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDOneClassSVM
from sklearn.mixture import GaussianMixture
from fe import *
def det_if(Z,seed): return -IsolationForest(n_estimators=300,max_samples=2048,random_state=seed,n_jobs=2).fit(Z).score_samples(Z)
def det_oc(Z,seed):
    ny=Nystroem(gamma=1/Z.shape[1]/ 4,n_components=400,random_state=seed).fit(Z[np.random.RandomState(seed).choice(len(Z),5000,replace=False)])
    P=ny.transform(Z)
    m=SGDOneClassSVM(nu=0.07,random_state=seed,max_iter=60,tol=None,fit_intercept=True).fit(P)
    return -m.decision_function(P)
def det_lof(Z,seed):
    # PCA to 20 dims first for speed
    P=PCA(25,random_state=seed).fit_transform(Z)
    return -LocalOutlierFactor(n_neighbors=40,n_jobs=2).fit(P).negative_outlier_factor_ if False else LocalOutlierFactor(n_neighbors=40,n_jobs=2).fit(P).negative_outlier_factor_*-1
def det_pca(Z,seed):
    p=PCA(15,random_state=seed).fit(Z); R=Z-p.inverse_transform(p.transform(Z)); return (R**2).sum(1)
def det_maha(Z,seed):
    C=np.cov(Z.T)+1e-3*np.eye(Z.shape[1]); Ci=np.linalg.inv(C); mu=Z.mean(0); X=Z-mu
    return np.einsum('ij,jk,ik->i',X,Ci,X)
def det_gmm(Z,seed):
    P=PCA(20,random_state=seed).fit_transform(Z)
    g=GaussianMixture(8,covariance_type='diag',random_state=seed,max_iter=100).fit(P); return -g.score_samples(P)
DET={'iforest':det_if,'ocsvm':det_oc,'lof':det_lof,'pca_recon':det_pca,'mahalanobis':det_maha,'gmm':det_gmm}
if __name__=='__main__':
    df=pd.read_csv('data/test.csv'); Z=prep(build(df))
    names=sys.argv[1:] or list(DET)
    for n in names:
        t=time.time(); s=DET[n](Z,0); np.save(f'out/{n}_score.npy',s); print(n,'done',round(time.time()-t,1),'s',flush=True)

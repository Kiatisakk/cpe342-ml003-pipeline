from common import *
from sklearn.preprocessing import StandardScaler,QuantileTransformer
from sklearn.kernel_approximation import Nystroem
from sklearn.svm import LinearSVC,SVC
from sklearn.linear_model import SGDClassifier,LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
def feats(d):
    X=d[INFO_NUM].copy()
    mi=X.isna().astype(float).values.sum(1,keepdims=True)
    for c in INFO_CAT:
        for v in sorted(tr[c].dropna().unique()): X[c+'_'+v]=(d[c]==v).astype(float)
    return X
def make(withid=False):
    A,B=feats(tr),feats(te)
    if withid: A['pid']=tr.pid;B['pid']=te.pid
    return A,B
def softmax(z):
    z=z-z.max(1,keepdims=True);e=np.exp(z);return e/e.sum(1,keepdims=True)
def prep(Xa,Xs):
    med=Xa.median();qt=QuantileTransformer(n_quantiles=200,output_distribution='normal',subsample=50000,random_state=0)
    Xa_=Xa.fillna(med);sc=StandardScaler().fit(qt.fit_transform(Xa_))
    f=lambda X:sc.transform(qt.transform(X.fillna(med)))
    return f(Xa),[f(X) for X in Xs]
def fp_linsvm(C):
    def fp(Xa,ya,Xs):
        Za,Zs=prep(Xa,Xs);m=LinearSVC(C=C,dual=False).fit(Za,ya)
        return [softmax(m.decision_function(Z)*2) for Z in Zs]
    return fp
def fp_nys(C,g,nc=1500):
    def fp(Xa,ya,Xs):
        Za,Zs=prep(Xa,Xs);ny=Nystroem(gamma=g,n_components=nc,random_state=0).fit(Za[:20000])
        m=LinearSVC(C=C,dual=False,max_iter=300).fit(ny.transform(Za),ya)
        return [softmax(m.decision_function(ny.transform(Z))*2) for Z in Zs]
    return fp
def fp_svc_sub(C,g,n=12000):
    def fp(Xa,ya,Xs):
        Za,Zs=prep(Xa,Xs);i=np.random.RandomState(0).choice(len(Za),n,replace=False)
        m=SVC(C=C,gamma=g,probability=False,decision_function_shape='ovr').fit(Za[i],ya[i])
        return [softmax(m.decision_function(Z)*2) for Z in Zs]
    return fp
if __name__=='__main__':
    import sys
    A,B=make(False)
    cv(fp_linsvm(0.1),'svm_linear_A',A,B,save=True)
    cv(fp_nys(1.0,0.01),'svm_nystroem_g.01_A',A,B,save=True)
    cv(fp_nys(0.3,0.003),'svm_nystroem_g.003_A',A,B,save=True)
    cv(fp_svc_sub(1.0,0.01),'svm_rbf_sub12k_A',A,B,save=True)

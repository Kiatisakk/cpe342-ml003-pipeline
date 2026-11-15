from svm import *
def fp_nys2(C,g,nc=300,n=30000):
    def fp(Xa,ya,Xs):
        Za,Zs=prep(Xa,Xs);i=np.random.RandomState(0).choice(len(Za),n,replace=False)
        ny=Nystroem(gamma=g,n_components=nc,random_state=0).fit(Za[i[:10000]])
        m=LinearSVC(C=C,dual=False,max_iter=200).fit(ny.transform(Za[i]),ya[i])
        return [softmax(m.decision_function(ny.transform(Z))*2) for Z in Zs]
    return fp
A,B=make(False)
import sys
if sys.argv[1]=='ny':
    cv(fp_nys2(0.5,0.01),'svm_nystroem300_g.01_A',A,B,save=True)
    cv(fp_nys2(0.5,0.003),'svm_nystroem300_g.003_A',A,B,save=True)
else:
    cv(fp_svc_sub(1.0,0.01,8000),'svm_rbf_sub8k_g.01_A',A,B,save=True)

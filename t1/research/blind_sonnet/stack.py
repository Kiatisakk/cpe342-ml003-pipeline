from common import *
from sklearn.linear_model import LogisticRegression
names=['gbm_eng_sig','et','nn_mlp','svm_spline_linear']
lg=lambda p:np.log(np.clip(p,1e-4,1-1e-4)/(1-np.clip(p,1e-4,1-1e-4)))
O_=np.column_stack([lg(np.load(O+n+'_oof.npy')) for n in names]);T_=np.column_stack([lg(np.load(O+n+'_test.npy')) for n in names])
def fp(A,ya,Bs):
    m=LogisticRegression(C=1.).fit(A,ya);return [m.predict_proba(x)[:,1] for x in Bs]
oof,r=run('stack_lr',fp,O_,T_,inner=3,save=False)
m=LogisticRegression(C=1.).fit(O_[lab],y[lab].astype(int));print(dict(zip(names,m.coef_[0].round(3))))
np.save(O+'stack_lr_oof.npy',oof);np.save(O+'stack_lr_test.npy',m.predict_proba(T_)[:,1])

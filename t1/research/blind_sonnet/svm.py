from common import *
import sys
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler,QuantileTransformer,SplineTransformer
from sklearn.impute import SimpleImputer
from sklearn.svm import LinearSVC,SVC
from sklearn.kernel_approximation import Nystroem,RBFSampler
from sklearn.linear_model import LogisticRegression,SGDClassifier
sig=lambda z:1/(1+np.exp(-z))
def prep(df):
    X=feats(df,SIG,True)
    # winsorised logs of heavy tailed
    for c in ['level_progression_speed','fr_age','rep_age','age_lo','age_hi','lv_lo','lv_hi','r']:
        X[c]=np.log1p(X[c].clip(lower=0))
    X['age_hi']=X.age_hi.clip(upper=8);X['age_lo']=X.age_lo.clip(upper=8)
    M=df[SIG].isna().astype(float).add_suffix('_na')
    return pd.concat([X,M],axis=1)
Xa=prep(tr);Xb=prep(te)
def base(final,**kw):
    return [SimpleImputer(strategy='median'),QuantileTransformer(n_quantiles=200,output_distribution='normal',subsample=50000,random_state=0) if kw.get('qt') else StandardScaler()]
def mk(kind,sub=None,**kw):
    def fp(A,ya,Bs):
        if sub and len(A)>sub:
            i=np.random.RandomState(0).choice(len(A),sub,replace=False);A2,y2=A[i],ya[i]
        else:A2,y2=A,ya
        steps=base(0,**kw)
        if kind=='lin':clf=LinearSVC(C=kw.get('C',.1),dual=False,max_iter=2000)
        elif kind=='nys':
            steps+= [Nystroem(gamma=kw.get('g',.02),n_components=kw.get('nc',400),random_state=0)]
            clf=LinearSVC(C=kw.get('C',.1),dual=False,max_iter=3000) if not kw.get('sgd') else SGDClassifier(loss='hinge',alpha=2e-5,max_iter=25,tol=None,random_state=0)
        elif kind=='rbf':clf=SVC(C=kw.get('C',1),gamma=kw.get('g','scale'),cache_size=1500)
        elif kind=='spl':
            steps+= [SplineTransformer(n_knots=8,degree=3,extrapolation='constant')]
            clf=LinearSVC(C=kw.get('C',.05),dual=False,max_iter=3000)
        m=make_pipeline(*steps,clf).fit(A2,y2)
        return [sig(m.decision_function(x)) for x in Bs]
    return fp
which=sys.argv[1]
V={'svm_linear':('lin',None,dict(C=.1)),
   'svm_spline_linear':('spl',None,dict(C=.05)),
   'svm_nystroem_g02':('nys',None,dict(g=.02,nc=300,sgd=1)),
   'svm_nystroem_g005':('nys',None,dict(g=.005,nc=500,C=.1)),
   'svm_nystroem_qt':('nys',None,dict(g=.02,nc=500,C=.1,qt=1)),
   'svm_rbf_sub10k':('rbf',10000,dict(C=1))}
for n in which.split(','):
    k,s,kw=V[n]
    run(n,mk(k,s,**kw),Xa.values,Xb.values,inner=3 if k!='rbf' else 2,save=True)

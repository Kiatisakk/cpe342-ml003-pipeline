from common import *
from svm import make,prep,softmax
from nb import nbX,fp_factory
from sklearn.preprocessing import SplineTransformer,StandardScaler,QuantileTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA,QuadraticDiscriminantAnalysis as QDA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier
import sys
which=sys.argv[1]
A,B=make(False)
def fp_lr(Xa,ya,Xs):
    Za,Zs=prep(Xa,Xs);st=SplineTransformer(n_knots=6,degree=3,knots='quantile').fit(Za[:30000,:len(INFO_NUM)])
    T=lambda Z:np.hstack([st.transform(Z[:,:len(INFO_NUM)]),Z[:,len(INFO_NUM):]])
    m=LogisticRegression(C=0.05,max_iter=300).fit(T(Za),ya);return [m.predict_proba(T(Z)) for Z in Zs]
def fp_lda(Xa,ya,Xs):
    Za,Zs=prep(Xa,Xs);m=LDA().fit(Za,ya);return [m.predict_proba(Z) for Z in Zs]
def fp_qda(Xa,ya,Xs):
    Za,Zs=prep(Xa,Xs);m=QDA(reg_param=0.1).fit(Za,ya);return [m.predict_proba(Z) for Z in Zs]
def fp_knn(Xa,ya,Xs):
    Za,Zs=prep(Xa,Xs);c=[i for i,n in enumerate(A.columns) if n in ('vip_tier','spending_frequency','avg_monthly_spending','total_spending_thb','rare_items_count','chat_activity_score','friend_count','play_frequency','avg_session_duration','peak_concurrent_hours')]+[i for i,n in enumerate(A.columns) if '_' in n and n.split('_')[0] in('engagement','loyalty')]
    m=KNeighborsClassifier(100,n_jobs=2,weights='distance').fit(Za[:,c],ya);return [m.predict_proba(Z[:,c]) for Z in Zs]
def fp_et(Xa,ya,Xs):
    med=Xa.median();m=ExtraTreesClassifier(120,min_samples_leaf=15,max_features=0.5,n_jobs=2,random_state=0).fit(Xa.fillna(med),ya);return [m.predict_proba(X.fillna(med)) for X in Xs]
def fp_rf(Xa,ya,Xs):
    med=Xa.median();m=RandomForestClassifier(100,min_samples_leaf=20,max_features=0.3,n_jobs=2,random_state=0).fit(Xa.fillna(med),ya);return [m.predict_proba(X.fillna(med)) for X in Xs]
if which=='lin':
    cv(fp_lda,'lda_A',A,B);cv(fp_qda,'qda_A',A,B);cv(fp_lr,'logreg_spline_A',A,B,save=True)
if which=='knn': cv(fp_knn,'knn100_A',A,B)
if which=='trees': cv(fp_et,'extratrees_A',A,B,save=True)
if which=='nb':
    n=tr.pid.values;ym=np.select([n<52000,n<84500,n<110500],[0,1,2],3)
    X,Xt=nbX(tr),nbX(te)
    for nb,sm,al in [(12,1.0,1),(20,1.0,1),(20,2.0,1),(30,2.0,1),(30,3.0,1)]:
        cv(fp_factory(nb,al,sm),'nb_%d_%.1f_A'%(nb,sm),X,Xt)

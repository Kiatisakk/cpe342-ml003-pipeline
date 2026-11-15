from prep import *
import sys
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,QuantileTransformer
from sklearn.pipeline import make_pipeline
Xa=prep(tr);Xb=prep(te)
def sk(make,sub=None):
    def fp(A,ya,Bs):
        if sub and len(A)>sub:
            i=np.random.RandomState(0).choice(len(A),sub,replace=False);A,ya=A[i],ya[i]
        m=make().fit(A,ya);return [m.predict_proba(x)[:,1] for x in Bs]
    return fp
imp=lambda:SimpleImputer(strategy='median')
nanfill=lambda:SimpleImputer(strategy='constant',fill_value=-1)
V={
 'et':(lambda:make_pipeline(nanfill(),ExtraTreesClassifier(200,min_samples_leaf=10,max_features=.5,n_jobs=4,random_state=0)),None,2),
 'rf':(lambda:make_pipeline(nanfill(),RandomForestClassifier(300,min_samples_leaf=10,max_features=.4,n_jobs=4,random_state=0)),None,2),
 'knn':(lambda:make_pipeline(imp(),QuantileTransformer(n_quantiles=200,output_distribution='normal',subsample=50000,random_state=0),KNeighborsClassifier(100,n_jobs=4)),20000,2),
 'nb':(lambda:make_pipeline(imp(),QuantileTransformer(n_quantiles=200,output_distribution='normal',subsample=50000,random_state=0),GaussianNB()),None,3),
}
for n in sys.argv[1].split(','):
    mkr,s,inn=V[n];run(n,sk(mkr,s),Xa.values,Xb.values,inner=inn,save=True)

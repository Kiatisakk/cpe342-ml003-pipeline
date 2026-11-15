from common import *
import common
from nb import nbX,fp_factory
n=tr.pid.values;ym=np.select([n<52000,n<84500,n<110500],[0,1,2],3)
X,Xt=nbX(tr),nbX(te)
def run(name,fp,save=False):
    t=time.time();oof=np.zeros((len(y),4));tp=np.zeros((len(Xt),4));sc=[]
    for a,b in FOLDS:
        o=fp(X[a],ym[a],[X[b],Xt]);oof[b]=o[0];tp+=o[1]/5;sc.append(mf1(y[b],o[0].argmax(1)))
    print('RESULT %s %.4f +- %.4f %.0fs'%(name,np.mean(sc),np.std(sc),time.time()-t))
    open(O+'results.tsv','a').write('%s\t%.4f\t%.4f\t%.0f\n'%(name,np.mean(sc),np.std(sc),time.time()-t))
    if save: np.save(O+name+'_oof.npy',oof);np.save(O+name+'_test.npy',tp)
for nb,sm in [(30,1.0),(40,1.0)]: run('nb_idcleanlabels_%d_%.1f_Aclean'%(nb,sm),fp_factory(nb,1.0,sm),save=(nb==30))

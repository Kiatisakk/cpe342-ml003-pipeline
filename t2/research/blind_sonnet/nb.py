from common import *
# histogram naive bayes on informative features (+ categorical engagement/loyalty), NaN skipped
NB_NUM=INFO_NUM;NBINS=24
def fp_factory(nb=NBINS,alpha=1.0,sm=1.0):
    def fp(Xa,ya,Xs):
        Xa=np.asarray(Xa);Xs=[np.asarray(X) for X in Xs]
        prior=np.bincount(ya,minlength=4)/len(ya);L=[np.log(prior)[None,:].repeat(len(X),0) for X in Xs]
        from scipy.ndimage import gaussian_filter1d
        for j in range(Xa.shape[1]):
            col=Xa[:,j];ok=~np.isnan(col)
            if j<len(NB_NUM):
                if len(np.unique(col[ok]))<=nb: edges=np.unique(col[ok]);edges=np.r_[edges[:-1]+np.diff(edges)/2] if len(edges)>1 else np.array([])
                else: edges=np.unique(np.quantile(col[ok],np.linspace(0,1,nb+1)[1:-1]))
                k=len(edges)+1
                f=lambda v:np.searchsorted(edges,v)
            else:
                k=int(np.nanmax(col))+1;f=lambda v:v.astype(int)
            cnt=np.zeros((4,k))
            for c in range(4):
                m=ok&(ya==c);cnt[c]=np.bincount(f(col[m]),minlength=k)
            if j<len(NB_NUM) and sm>0: cnt=gaussian_filter1d(cnt,sm,axis=1,mode='nearest')
            pr=(cnt+alpha)/(cnt+alpha).sum(1,keepdims=True)
            for i,X in enumerate(Xs):
                v=X[:,j];o=~np.isnan(v)
                L[i][o]+=np.log(pr[:,f(v[o])]).T
        out=[]
        for l in L:
            l=l-l.max(1,keepdims=True);p=np.exp(l);out.append(p/p.sum(1,keepdims=True))
        return out
    return fp
def nbX(d,withid=False):
    X=d[NB_NUM].copy()
    for c in INFO_CAT: X[c]=pd.Categorical(d[c],categories=sorted(tr[c].dropna().unique())).codes.astype(float);X.loc[X[c]<0,c]=np.nan
    return X.values
if __name__=='__main__':
    A,B=nbX(tr),nbX(te)
    for nb,sm in [(20,1.0),(40,1.5)]:
        cv(fp_factory(nb,1.0,sm),'nb_hist_%d_%.1f_A'%(nb,sm),A,B,save=(nb==40))

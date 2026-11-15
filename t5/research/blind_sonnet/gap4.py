import warnings;warnings.filterwarnings('ignore')
from feat import *
out={}
for m in metrics:
    if m in('login_lat','login_lon','device_count','refund_count','guild_changes','ip_hash_entropy'):continue
    X=d[[f'{m}_{i}' for i in range(1,5)]].values
    b=np.nanmean(X[:,:3],1)
    if m=='skill_trajectory': out[m]=X[:,3]-b
    else: out[m]=np.log((X[:,3]+1e-3*np.nanstd(X[:,3]))/(b+1e-3*np.nanstd(X[:,3]))) if m not in() else 0
    # for ratio use log ratio with eps
D=pd.DataFrame(out)
D.to_pickle('out/D4.pkl')
for k in D.columns:
    v=D[k].dropna().values; s=np.sort(v); n=len(s); g=np.diff(s)
    ref=np.median(g[int(.3*n):int(.7*n)])
    for side,rng in (('low',range(0,int(.12*n))),('high',range(int(.88*n),n-1))):
        rng=list(rng); j=rng[int(np.argmax(g[rng]))]
        print(f'{k:26s}{side:5s} ratio={g[j]/ref:8.0f} gap {s[j]:.3f}->{s[j+1]:.3f} tail_n={(j+1) if side=="low" else n-j-1}')

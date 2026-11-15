import warnings;warnings.filterwarnings('ignore')
from feat import *
cols={}
for m in metrics:
    if m in('login_lat','login_lon','device_count','refund_count','guild_changes'):continue
    X=d[[f'{m}_{i}' for i in range(1,4)]].values
    cols[m+'_mean123']=np.nanmean(X,1)
for c in ['account_age_days','behavior_stability']: cols[c]=d[c].values
res=[]
for k,v in cols.items():
    ok=~np.isnan(v); idx=np.where(ok)[0]; o=np.argsort(v[ok]); s=v[ok][o]; n=len(s)
    g=np.diff(s)
    for side,rng in (('low',range(0,int(.15*n))),('high',range(int(.85*n),n-1))):
        rng=list(rng); gg=g[rng]
        # local scale: median gap in the central 30-70% region
        ref=np.median(g[int(.3*n):int(.7*n)])
        j=rng[int(np.argmax(gg))]
        res.append((k,side,gg.max()/ref,s[j],s[j+1],(j+1) if side=='low' else n-j-1))
res.sort(key=lambda r:-r[2])
for r in res[:30]: print(f'{r[0]:36s}{r[1]:5s} gapratio={r[2]:8.1f} between {r[3]:.4g} and {r[4]:.4g}  tail_n={r[5]}')

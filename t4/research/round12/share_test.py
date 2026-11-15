# Are the LB digit shares compatible with truth = some relabelling of the real test labels (any row order)?
import pandas as pd,numpy as np,itertools
from scipy.optimize import minimize
P='submissions/probes/t4_constbase/'; D='C:/Users/kiati/Downloads/'
pr=[(P+'4_A1_mix7_v12c.csv',.27935,.27401),(P+'5_B1_ints.csv',.28420,.27401),(P+'7_M1_mapping_01234_object.csv',.30552,.27401),
(P+'8_M2b_mapping.csv',.31337,.27401),(P+'31024.csv',.31336,.27401),(P+'41203.csv',.31254,.27401),(P+'sortedids_datacard.csv',.31327,.27401),
(P+'12_S1_rowblocks.csv',.31307,.27401),(P+'14_ISO_fiveN_is_1.csv',.28336,.27401),(D+'sample_submission_ding.csv',.37399,.35559)]
X=[]
for f,lb,b in pr:
    t=pd.read_csv(f,dtype=str,keep_default_na=False).task4; vc=t.value_counts(normalize=True)
    p=np.array([vc.get(str(d),0.) for d in range(5)]); den=len(set(vc.index)|set('01234')); X.append((p,den,lb-b)); print(f.split('/')[-1][:28],p.round(3),den,round(lb-b,5))
def pred(s,w): return np.array([w*np.sum(2*p*s/np.maximum(p+s,1e-12))/den for p,den,_ in X])
y=np.array([x[2] for x in X]); rm=lambda s,w:np.sqrt(np.mean((pred(s,w)-y)**2))
def fit(free_w):
    f=lambda v:rm(np.exp(v[:5])/np.exp(v[:5]).sum(),v[5] if free_w else .2)
    r=minimize(f,np.r_[np.log([.14,.3,.24,.21,.11]),.2],method='Nelder-Mead',options=dict(maxiter=20000,maxfev=20000,xatol=1e-6,fatol=1e-9))
    return r.fun,(np.exp(r.x[:5])/np.exp(r.x[:5]).sum()).round(4),r.x[5] if free_w else .2
print('free shares w=.2',fit(False)); print('free shares free w',fit(True))
C=['7-11k','fiveN','freefried','pube','roblock']
te=np.array([.15157,.25969,.15169,.25911,.17795]); tr=np.array([.394,.1397,.1774,.18785,.10106])
def perms(mix,lab,show=3):
    out=[]
    for q in itertools.permutations(range(5)):
        s=mix[list(q)]
        for w in np.arange(.15,.2601,.0025): out.append((rm(s,w),w,[C[i] for i in q]))
    out.sort(key=lambda r:r[0]); print(lab); [print(' ',round(a*1e5)/1e5,round(w,4),q) for a,w,q in out[:show]]
perms(te,'truth mix = test mix, any mapping, free w')
perms(tr,'truth mix = train mix, any mapping, free w',1)
al=(31546*tr+25889*te)/57435; perms(al,'truth mix = train+test pool',2)
print('noise: LB sd of a chance probe by public size')
rng=np.random.default_rng(0); s=np.array([.136,.298,.24,.211,.115])
y4=pd.read_csv(P+'8_M2b_mapping.csv',dtype=str).task4.values
for n in [500,1000,2500,7767,12945,25889]:
    v=[]
    for _ in range(300):
        i=rng.choice(25889,n,replace=False); a=y4[i]; b=rng.choice(list('01234'),n,p=s/s.sum())
        v.append(np.mean([2*np.sum((a==d)&(b==d))/max(np.sum(a==d)+np.sum(b==d),1) for d in '01234'])*.2)
    print(n,round(np.std(v),5),'sd of digit-1 share under .26:',round(np.sqrt(.26*.74*(1-n/25889)/n),4))

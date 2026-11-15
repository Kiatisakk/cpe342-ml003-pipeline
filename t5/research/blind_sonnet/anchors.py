import warnings;warnings.filterwarnings('ignore')
from feat import *
pd.set_option('display.width',250,'display.max_rows',200)
D=pd.read_pickle('out/D4.pkl');g3=np.load('out/g3.npy');g4=np.load('out/g4.npy');F=pd.read_pickle('out/stat_flags.pkl')
C0=((D.skill_rating>0.26)&(D.skill_trajectory>25)).values
B0=((D.play_pattern_consistency<-0.3)&g4).values
A0=g3&g4
# absolute change features (p4 - base) and ratio
def X(m,i):return d[f'{m}_{i}'].values
rows=[]
for m in D.columns:
    b=np.nanmean(np.c_[X(m,1),X(m,2),X(m,3)],1); x4=X(m,4)
    r=np.log((x4+1e-6+0.05*np.nanstd(x4))/(b+1e-6+0.05*np.nanstd(x4)))
    out=~(A0|B0|C0|F.any(axis=1).values|g3|g4)
    for nm,mk in (('A',A0),('B',B0),('C',C0)):
        v=r[mk&~np.isnan(r)]; o=r[out&~np.isnan(r)]
        rows.append((m,nm,len(v),np.percentile(v,5),np.median(v),np.percentile(v,95),np.percentile(o,0.05),np.percentile(o,99.95),np.nanmin(o),np.nanmax(o)))
R=pd.DataFrame(rows,columns=['m','grp','n','p5','med','p95','out_p.05','out_p99.95','out_min','out_max'])
R['sep']=np.where(R['med']>0,(R['p5']-R['out_max']),(R['out_min']-R['p95']))
print(R[(R.sep>0)].round(3).to_string())
print('-----')
R['effect']=np.where(R['med']>0,R['p5']-R['out_p99.95'],R['out_p.05']-R['p95'])
print(R.sort_values('effect',ascending=False).head(30).round(3).to_string())

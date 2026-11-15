import warnings;warnings.filterwarnings('ignore')
from feat import *
pd.set_option('display.width',250,'display.max_rows',200)
def X(m,i):return d[f'{m}_{i}'].values
def b123(m):return np.nanmean(np.c_[X(m,1),X(m,2),X(m,3)],1)
def lr(m):
    s=0.05*np.nanstd(X(m,4)); return np.log((X(m,4)+s)/(b123(m)+s))
L=lambda k,i:X('login_'+k,i)
hl=np.nanmean([L('lat',1),L('lat',2)],0);ho=np.nanmean([L('lon',1),L('lon',2)],0)
j3=np.nan_to_num(hav(hl,ho,L('lat',3),L('lon',3))); j4=np.nan_to_num(hav(hl,ho,L('lat',4),L('lon',4)))
g3=(j3>5)|(np.nan_to_num(L('lat',3))>90)|(np.nan_to_num(L('lat',3))<-90)
g4=(j4>5)|(np.abs(np.nan_to_num(L('lat',4)))>90)
sk=np.nan_to_num(lr('skill_rating'))>0.26
st=np.nan_to_num(X('skill_trajectory',4)-b123('skill_trajectory'))>25
C=sk|st
pf=np.nan_to_num(lr('purchase_frequency'))>0.9
pam=np.nan_to_num(lr('purchase_amount_mean'))>0.55
play=np.nan_to_num(lr('play_pattern_consistency'))<-0.3
A=g3|(pf&pam)|(g4&(pf|pam)&~play)
B=(play|g4)&~A&~C
G={'T_geoA(g3 jump + purchase spike)':A,'T_geoB(play drop/ geo4)':B,'S_perfspike':C}
F=pd.read_pickle('out/stat_flags.pkl')
bot=(F.ipent|F.ltv|F.sessstd|F.kdastd).values
stat={'S_bot':bot,'S_lowcomm':F.comm.values,'S_lowact':F.act.values,'S_lowbeh':F.beh.values}
allg={**G,**stat}
for k,v in allg.items():print(k,v.sum())
M=np.c_[[v for v in allg.values()]].T
print('overlap rows (in >1 group):',(M.sum(1)>1).sum(),' union',M.any(1).sum())
print('A parts: g3',g3.sum(),'pf&pam',(pf&pam).sum(),'g4&(pf|pam)&~play',(g4&(pf|pam)&~play).sum(),' pf',pf.sum(),'pam',pam.sum())
print('B parts: play',play.sum(),'g4',g4.sum(),'g4-only(B)',(g4&~play&~A&~C).sum())
Z4=pd.read_pickle('out/Z4.pkl'); nb=(Z4.abs()>6).sum(1).values
left=(~M.any(1))
print('leftover rows with >=2 big p4 shifts:',(left&(nb>=2)).sum(),' >=3',(left&(nb>=3)).sum(),'>=1',(left&(nb>=1)).sum())
np.save('out/rule_M.npy',M); pd.Series(list(allg)).to_csv('out/rule_names.csv',index=False)
# leftover signature
cols=list(Z4.columns);sub=Z4[left&(nb>=2)]
print(((sub.abs()>6)*np.sign(sub)).fillna(0).apply(lambda r:' '.join(f"{'+' if v>0 else '-'}{c}" for c,v in r.items() if v!=0),axis=1).value_counts().head(15))

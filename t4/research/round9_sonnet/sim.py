import pandas as pd, numpy as np
from sklearn.metrics import f1_score
rng=np.random.default_rng(0)
d=pd.read_csv('submissions/FINAL/probe_v12t4_UNION7formats_T1r60.csv',dtype=str)
g=pd.read_csv('t4/research/probe_v12t4_groups.csv').group.values
p=pd.read_csv('t4/preds/v3_5f/preds/task4_v3_5f.csv')
cls=p.pred_num.values.astype(int)   # datacard order proxy truth
N=len(d)
names={'int_datacard':lambda c:str(c),'int1_datacard':lambda c:str(c+1),
 'int_overview':lambda c:str([4,3,2,0,1][c]) ,
 'int1_overview':lambda c:str([4,3,2,0,1][c]+1),
 'train':lambda c:['7-11k','fiveN','freefried','pube','roblock'][c],
 'train711K':lambda c:['7-11K','fiveN','freefried','pube','roblock'][c],
 'display':lambda c:['7-Eleven Knight','FiveN','Free Fried','PUBE','Roblock'][c],
 'lower':lambda c:['7-11k','fiven','freefried','pube','roblock'][c],
 'upper':lambda c:['7-11K','FIVEN','FREEFRIED','PUBE','ROBLOCK'][c],
 'card_names':lambda c:['7-11 Knights','FiveN','Free Fried','PUBE','Roblock'][c],
 'canva_711K':lambda c:['7-11K','fiveN','freefried','pube','roblock'][c],
 'pubemobile':lambda c:['7-11 Knights','FiveN','Free Fried','PUBE Mobile','Roblock'][c]}
# overview order: 0 pube,1 freefried,2 fiveN,3 roblock,4 7-11k ; datacard c -> overview idx: 7-11k->4,fiveN->2,freefried->1,pube->0,roblock->3
names['int_overview']=lambda c:str([4,2,1,0,3][c]); names['int1_overview']=lambda c:str([4,2,1,0,3][c]+1)
pred=d.task4.values
def mf(yt,yp,mode):
    if mode=='union': return f1_score(yt,yp,average='macro',zero_division=0)
    if mode=='true': 
        L=sorted(set(yt)); return f1_score(yt,yp,labels=L,average='macro',zero_division=0)
res=[]
for k,f in names.items():
    yt=np.array([f(c) for c in cls])
    row={'enc':k}
    for mode in ['union','true']:
        row[mode+'_ordered']=mf(yt,pred,mode)
        sh=[mf(rng.permutation(yt),pred,mode) for _ in range(3)]
        row[mode+'_shuf']=np.mean(sh)
    res.append(row)
r=pd.DataFrame(res).round(4); print(r.to_string())
r.to_csv('t4/research/round9_sonnet/sim_mix.csv',index=False)

import pandas as pd, numpy as np, itertools
from sklearn.metrics import f1_score
d=pd.read_csv('submissions/FINAL/probe_v12t4_UNION7formats_T1r60.csv',dtype=str)
pred=d.task4.values
p=pd.read_csv('t4/preds/v3_5f/preds/task4_v3_5f.csv'); c=p.pred_num.values.astype(int)
ident=np.arange(5); ov=np.array([4,2,1,0,3])  # datacard idx -> overview int
out=[]
for perm in itertools.permutations(range(5)):
    perm=np.array(perm)  # truth int for datacard class k = perm[k]
    pure_dc=(perm==ident).sum(); pure_ov=(perm==ov).sum()
    s=f1_score(perm[c].astype(str),pred,average='macro',zero_division=0)
    s1=f1_score((perm[c]+1).astype(str),pred,average='macro',zero_division=0)
    out.append((tuple(perm),pure_dc,pure_ov,round(s,4),round(s1,4)))
o=pd.DataFrame(out,columns=['perm','fix_datacard','fix_overview','mix_0idx','mix_1idx'])
ok=o[(o.fix_datacard==0)&(o.fix_overview==0)]
print(len(o),len(ok)); print(ok.mix_0idx.describe()); print(ok.sort_values('mix_0idx').to_string())
print('1-indexed perms (with identity-ish 1..5 untested pure): mix_1idx stats'); print(o.mix_1idx.describe())
o.to_csv('t4/research/round9_sonnet/perm_table.csv',index=False)

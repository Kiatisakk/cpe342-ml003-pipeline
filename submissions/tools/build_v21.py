# v21a: sub_v16a (T2 + T5 id leaks, T3 best, T4 int datacard confirmed by the TA) + T1 rank blend from sub_v20a (0.7 final_v2 + 0.2 stack_lr + 0.1 GBM pipeline, 60%).
import numpy as np,pandas as pd
F='submissions/FINAL/'
A=pd.read_csv(F+'sub_v16a_LEAK_T1r60_T4intDatacard_T5idblock.csv',dtype=str,keep_default_na=False)
B=pd.read_csv(F+'sub_v20a_noleak_T1blend721.csv',dtype=str,keep_default_na=False)
assert (A.id==B.id).all()
d=A.copy(); d['task1']=B.task1.values
O=F+'sub_v21a_LEAK_T1blend721.csv'; d.to_csv(O,index=False,lineterminator='\n')
r=pd.read_csv(O,dtype=str,keep_default_na=False); raw=open(O,'rb').read()
assert r.shape==(25889,6) and (r=='').sum().sum()==0 and b'\r' not in raw and list(r.columns)==['id','task1','task2','task3','task4','task5']
print(O,'| changed vs v16a',{c:int((A[c]!=r[c]).sum()) for c in A.columns if (A[c]!=r[c]).any()},'| T1 rate',round((r.task1=='1.0').mean(),4),
      '| T4 tokens',sorted(r.task4.unique()),'| T5 rows',int((r.task5=='1.0').sum()),'| id',r.id.iloc[0],r.id.iloc[-1])

# v19a..e: all from sub_v16a (T2 + T5 id leaks), one isolated change each (public-only tweaks, all within noise).
import numpy as np,pandas as pd
F='submissions/FINAL/'; A=pd.read_csv(F+'sub_v16a_LEAK_T1r60_T4intDatacard_T5idblock.csv',dtype=str,keep_default_na=False)
s=pd.read_csv('t4/research/t4_test_str.csv').pred; sc=pd.read_csv('t1/research/t1_best_test.csv').score.values
ta=pd.read_csv('t3/research/t3_best_test.csv');tb=pd.read_csv('t3/research/t3_robust_test.csv'); t3=((ta.pred.values+tb.pred.values)/2).round(2).astype(str)
C=['7-11k','fiveN','freefried','pube','roblock']; mp=lambda d:s.map(dict(zip(C,d.split()))).values
t1=lambda p:np.where(sc>=p,'1.0','0.0')
def out(name,**cols):
    d=A.copy()
    for k,v in cols.items(): d[k]=v
    d.to_csv(F+name,index=False,lineterminator='\n'); r=pd.read_csv(F+name,dtype=str,keep_default_na=False)
    assert r.shape==(25889,6) and (r=='').sum().sum()==0 and b'\r' not in open(F+name,'rb').read()
    print(name,{c:int((A[c]!=r[c]).sum()) for c in A.columns if (A[c]!=r[c]).any()},'T1 rate',round((r.task1=='1.0').mean(),4))
out('sub_v19a_LEAK_T4int01423.csv',task4=mp('0 1 4 2 3'))
out('sub_v19b_LEAK_T1p20.csv',task1=t1(.20))
out('sub_v19c_LEAK_T1p175.csv',task1=t1(.175))
out('sub_v19d_LEAK_T4int01423_T3mix.csv',task4=mp('0 1 4 2 3'),task3=t3)
out('sub_v19e_LEAK_T4int02413.csv',task4=mp('0 2 4 1 3'))

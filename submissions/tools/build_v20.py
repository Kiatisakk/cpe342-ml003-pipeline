# v20a..e: no id leak. Base sub_v18a (T2 blend+affine, T5 box + lowstab-NaN post>=0.10).
# T1 = rank blend of final_v2 (best_test_proba) + blind stack_lr + GBM pipeline (XGB/LGB/Cat); fully nested CV +0.00068 F2, 14/15 folds up.
import numpy as np,pandas as pd
from scipy.stats import rankdata
F='submissions/FINAL/'; A=pd.read_csv(F+'sub_v18a_noleak_T2blendAffine.csv',dtype=str,keep_default_na=False)
R=lambda v:rankdata(v)/len(v)
o=pd.read_csv('t1/research/t1_best_test.csv'); p=pd.read_csv('t1/preds/t1_pipeline/t1_blend.csv'); s=np.load('t1/research/blind_sonnet/stack_lr_test.npy')
assert (o.id==A.id).all() and (p.id==A.id).all() and len(s)==len(A)
def t1(ws,wp,rate=.60):
    b=(1-ws-wp)*R(o.score.values)+ws*R(s)+wp*R(p.score.values); return np.where(b>=np.quantile(b,1-rate),'1.0','0.0')
post=pd.read_csv('t5/research/out/lowstab_nan_scores.csv').post_lowstab_nan.values
box=pd.read_csv('t5/research/candidates/t5_union_plus_acckda_box.csv'); bx=box[[c for c in box.columns if c!='id'][0]].astype(float).values==1
t5=lambda th:np.where(bx|(post>=th),'1.0','0.0')
assert (t5(.1)==A.task5.values).all()   # v18a T5 = box + post>=0.10
def out(name,**cols):
    d=A.copy()
    for k,v in cols.items(): d[k]=v
    d.to_csv(F+name,index=False,lineterminator='\n'); r=pd.read_csv(F+name,dtype=str,keep_default_na=False)
    assert r.shape==(25889,6) and (r=='').sum().sum()==0 and b'\r' not in open(F+name,'rb').read()
    print(name,{c:int((A[c]!=r[c]).sum()) for c in A.columns if (A[c]!=r[c]).any()},'T1 rate',round((r.task1=='1.0').mean(),4),'T5 rows',int((r.task5=='1.0').sum()))
out('sub_v20a_noleak_T1blend721.csv',task1=t1(.2,.1))
out('sub_v20b_noleak_T1blend811.csv',task1=t1(.1,.1))
out('sub_v20c_noleak_T1blend721_r61.csv',task1=t1(.2,.1,.61))
out('sub_v20d_noleak_T1blend721_T5post05.csv',task1=t1(.2,.1),task5=t5(.05))
out('sub_v20e_noleak_T1blend721_T5post15.csv',task1=t1(.2,.1),task5=t5(.15))

# v23a: sub_v21a + T4 = blend of v3_5f ConvNeXt-T DINOv3 (0.6) + v4 EVA02-S (0.4, friend's 5-fold Kaggle run, TTA plain+hflip+up).
# Nested-CV OOF macro-F1 0.99170 vs v3 alone 0.98886 (t4/research/t4_blend_runs.py). T4 int text, datacard order (TA-confirmed).
import pandas as pd
F='submissions/FINAL/'
A=pd.read_csv(F+'sub_v21a_LEAK_T1blend721.csv',dtype=str,keep_default_na=False)
b=pd.read_csv('t4/research/blend_eva_plain_hflip_up/t4_blend.csv')
v3=pd.read_csv('t4/preds/v3_5f/preds/task4_v3_5f.csv')
assert (b.id.values==A.id.values).all() and (b.id_img.values==v3.id_img.values).all()
assert (A.task4.values==v3.pred_num.astype(str).values).all()   # v21a T4 = v3 datacard ints
d=A.copy(); d['task4']=b.pred_num.astype(int).astype(str).values
O=F+'sub_v23a_LEAK_T4blendEva02.csv'; d.to_csv(O,index=False,lineterminator='\n')
r=pd.read_csv(O,dtype=str,keep_default_na=False)
assert r.shape==(25889,6) and (r=='').sum().sum()==0 and b'\r' not in open(O,'rb').read()
print(O,'| changed vs v21a',{c:int((A[c]!=r[c]).sum()) for c in A.columns if (A[c]!=r[c]).any()},'| T4 tokens',sorted(r.task4.unique()))

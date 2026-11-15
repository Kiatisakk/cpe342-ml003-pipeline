# Builds sub_v18a..e from v16a (leak) / v16b (no leak). Only CV-backed changes or final-selection hedges, no probes.
import numpy as np,pandas as pd
from sklearn.metrics import f1_score
F='submissions/FINAL/'; rd=lambda n:pd.read_csv(F+n,dtype=str,keep_default_na=False)
A=rd('sub_v16a_LEAK_T1r60_T4intDatacard_T5idblock.csv'); B=rd('sub_v16b_noleak_T1r60_T4intDatacard_T5lsnan.csv')
# T2 blend + affine (fixed-point F1 rule fitted on full OOF; nested CV +0.00088 +- 0.00097)
y2=pd.read_csv('cpe342-kaggle-main/02_multiclass_classification/train.csv').segment.values; S='t2/research/blind_sonnet/'
Bo=.87*np.load('t2/research/best_oof_proba.npy')+.08*np.load(S+'logreg_spline_A_oof.npy')+.05*np.load(S+'mlp_torch_A_oof.npy')
Bt=.87*np.load('t2/research/best_test_proba.npy')+.08*np.load(S+'logreg_spline_A_test.npy')+.05*np.load(S+'mlp_torch_A_test.npy')
pi=np.bincount(y2,minlength=4)/len(y2);q=pi.copy();Fk=np.full(4,.75)
for _ in range(8):
    a=2/(pi+q);b=-Fk/(pi+q);pr=(a*Bo+b).argmax(1);q=np.bincount(pr,minlength=4)/len(y2)
    Fk=np.array([2*((pr==k)&(y2==k)).mean()/(pi[k]+q[k]) for k in range(4)])
print('T2 OOF F1m plain',round(f1_score(y2,np.load('t2/research/best_oof_proba.npy').argmax(1),average='macro'),5),'blend+affine',round(f1_score(y2,pr,average='macro'),5))
t2=((a*Bt+b).argmax(1)).astype(float).astype(str)
# T3 50/50 with robust model (OOF tie, lower variance)
ta=pd.read_csv('t3/research/t3_best_test.csv');tb=pd.read_csv('t3/research/t3_robust_test.csv')
assert (ta.id==A.id).all() and (tb.id==A.id).all()
t3=((ta.pred.values+tb.pred.values)/2).round(2).astype(str)
# T4 overview order
s=pd.read_csv('t4/research/t4_test_str.csv'); ov=s.pred.map({'pube':'0','freefried':'1','fiveN':'2','roblock':'3','7-11k':'4'}).values
# T5: id block plus box-only rows (box 2055 rows minus block)
t5test=pd.read_csv('cpe342-kaggle-main/05_anomaly_detection/test.csv'); box=pd.read_csv('t5/research/candidates/t5_union_plus_acckda_box.csv')
bc=[c for c in box.columns if c!='id'][0]; boxpos=box[bc].astype(float).values==1; blk=A.task5.values=='1.0'
t5=np.where(blk|boxpos,'1.0','0.0'); print('box rows',boxpos.sum(),'box outside block',(boxpos&~blk).sum())
def out(base,name,**cols):
    d=base.copy()
    for k,v in cols.items(): d[k]=v
    d.to_csv(F+name,index=False,lineterminator='\n'); r=pd.read_csv(F+name,dtype=str,keep_default_na=False)
    assert r.shape==(25889,6) and (r=='').sum().sum()==0 and b'\r' not in open(F+name,'rb').read()
    print(name,'| changed vs base',{c:int((base[c]!=r[c]).sum()) for c in base.columns if c!='id' and (base[c]!=r[c]).any()})
out(B,'sub_v18a_noleak_T2blendAffine.csv',task2=t2)
out(B,'sub_v18b_noleak_T2blendAffine_T4overview.csv',task2=t2,task4=ov)
out(B,'sub_v18c_noleak_T2blendAffine_T3mix.csv',task2=t2,task3=t3)
out(A,'sub_v18d_LEAK_T3mix.csv',task3=t3)
out(A,'sub_v18e_LEAK_T5blockPlusBox.csv',task5=t5)

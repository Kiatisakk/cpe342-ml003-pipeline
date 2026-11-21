# T5 NEW: expected F3 under the generator accounting (2,070 anomalies: 2,052 certain in our file, 1 lowact row with all activity cells NaN,
# 17 hidden lowstab rows among 128 candidates) and candidate prediction files. Nothing is submitted.
import pandas as pd, numpy as np
cur=pd.read_csv("t5/preds/v2data/t5v2_union_box_lsnan.csv"); col=cur.columns[1]; pred=cur[col].values==1
post=pd.read_csv("work/t5v2/out/lowstab_nan_scores.csv").post_lowstab_nan.values
c=pd.read_csv("work/insight/e06_lowstab_candidates.csv"); ci=c.row.values
ROW=25766; P_ROW=0.28; H=17; N=2070
odds=post[ci]/(1-post[ci]+1e-12); lo,hi=.1,50
for _ in range(80):
    k=(lo+hi)/2; lo,hi=(k,hi) if (k*odds/(1+k*odds)).sum()<H else (lo,k)
p17=np.zeros(len(cur)); p17[ci]=k*odds/(1+k*odds)
certain=pred.copy(); certain[ci]=False; print("certain rows in the current file:",certain.sum(),"| lsnan rows:",(pred[ci]).sum(),"| odds multiplier %.3f"%k)
def ef3(sel):
    tp=certain[sel].sum()+p17[sel].sum()+(P_ROW if sel[ROW] else 0); fp=sel.sum()-tp; fn=N-tp; return 10*tp/(10*tp+9*fn+fp),tp,fp,fn
out={}
A=pred.copy(); out["current (2,104)"]=A
B=A.copy(); B[ROW]=True; out["+ lowact row"]=B
C=B.copy(); C[ci]=p17[ci]>=.0995; out["+ lowact row, lsnan prior 17"]=C
D=B.copy(); D[ci]=False; D[ci[np.argsort(-p17[ci])[:17]]]=True; out["+ lowact row, top-17 lsnan only"]=D
E=B.copy(); E[ci]=True; out["+ lowact row, all 128 candidates"]=E
for nm,s in out.items():
    f,tp,fp,fn=ef3(s); print(f"{nm:36s} rows {s.sum():5d}  E[TP] {tp:7.1f} E[FP] {fp:5.1f} E[FN] {fn:5.1f}  E[F3] {f:.5f}")
for nm,s in (("t5v2_union_box_lsnan_la",B),("t5v2_union_box_lsnan17_la",C)):
    d=cur.copy(); d[col]=s.astype(int); d.to_csv(f"t5/preds/v2data/{nm}.csv",index=False,lineterminator="\n"); print("wrote",nm,int(s.sum()),"rows; changed vs current:",int((s!=pred).sum()))

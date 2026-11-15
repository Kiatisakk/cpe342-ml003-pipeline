# v25 (12 Nov, data re-release): T2 + T5 datasets regenerated, T4 test.csv re-ordered (ANS id -> file_name).
# Base = sub_v24b (no leak). T1, T3 data unchanged -> kept.
# T2 = same recipe on new data (work/t2v2/make_t2v2.py); T4 = v3/EVA02 blend realigned by file name; T5 = pipeline + box + lsnan on new test.
import pandas as pd
F = 'submissions/FINAL/'
B = pd.read_csv(F + 'sub_v24b_noleak_T3seedbag10_T4blendEva02.csv', dtype=str, keep_default_na=False)
t2 = pd.read_csv('t2/preds/v2data/t2v2_blend_affine.csv')
t4 = pd.read_csv('t4/preds/realigned_v2/t4_blend_v3eva02.csv')
t5 = pd.read_csv('t5/preds/v2data/t5v2_union_box_lsnan.csv')
for d in (t2, t4, t5): assert (d.id.values == B.id.values).all()
d = B.copy()
d['task2'] = t2.pred.astype(float).astype(str).values      # same float-text format as v18a
d['task4'] = t4.pred_num.astype(int).astype(str).values    # int text, datacard order
d['task5'] = t5.pred.astype(float).astype(str).values
# keep v24b's T5 token format
if not B.task5.str.contains(r'\.').any(): d['task5'] = t5.pred.astype(int).astype(str).values
O = F + 'sub_v25a_newdata_T2v2_T4realigned_T5v2.csv'
d.to_csv(O, index=False, lineterminator='\n')
r = pd.read_csv(O, dtype=str, keep_default_na=False)
print(O, r.shape, '| changed vs v24b', {c: int((B[c] != r[c]).sum()) for c in B.columns if (B[c] != r[c]).any()})
print('tokens', {c: sorted(r[c].unique())[:6] for c in ['task2', 'task4', 'task5']}, '| T5 n', int(r.task5.astype(float).sum()))

# Swap a T4 blend into sub_v25a. The blend csv is in the OLD test.csv order (id_img column), so it is realigned to the
# new test.csv (ANS id -> file_name) first.
# usage: python submissions/tools/build_v26_t4.py <blend_dir with t4_blend.csv> <output name suffix>
#   e.g. python submissions/tools/build_v26_t4.py t4/research/blend_coloraug coloraug
import sys, pandas as pd
src, tag = sys.argv[1], sys.argv[2]
F = 'submissions/FINAL/'
B = pd.read_csv(F + 'sub_v25a_newdata_T2v2_T4realigned_T5v2.csv', dtype=str, keep_default_na=False)
new = pd.read_csv('cpe342-kaggle-main/04_image_classification/test.csv')
f = new.file_name.str.replace('.jpg', '', regex=False).values
b = pd.read_csv(f'{src}/t4_blend.csv').set_index('id_img').loc[f]
assert (new.id.values == B.id.values).all() and b.pred_num.notna().all()
d = B.copy(); d['task4'] = b.pred_num.astype(int).astype(str).values
O = F + f'sub_v26_{tag}.csv'; d.to_csv(O, index=False, lineterminator='\n')
print(O, '| T4 rows changed vs v25a:', int((d.task4 != B.task4).sum()), '| share', d.task4.value_counts(normalize=True).sort_index().round(3).to_dict())

# v22a/b: sub_v21a + T1 pseudo-label effect (13,184 UNLABELED TRAIN rows, no test rows in training).
# The final_v2 component of the v21a T1 rank blend gets the pseudo-label shift added:
#   R(final_v2) + R(pseudo variant) - R(same-recipe baseline)   (both from round13_build_test.py, lr .05, seeds 42/7/2024, Kaggle kernel task1-pseudo-r13)
# so only the pseudo-label effect changes. Then 0.7 / 0.2 stack_lr / 0.1 GBM pipeline, top 60%, as v20a/v21a.
# Seed-42 nested binary F2: soft_w05 +0.00041 (4/5), hard +0.00037 (3/5); test-weighted -0.00006 / -0.00029 (noise).
import numpy as np, pandas as pd
from scipy.stats import rankdata
F = 'submissions/FINAL/'; Q = 't1/preds/pseudo_r13/out/'
A = pd.read_csv(F + 'sub_v21a_LEAK_T1blend721.csv', dtype=str, keep_default_na=False)
R = lambda v: rankdata(v) / len(v)
o = pd.read_csv('t1/research/t1_best_test.csv'); p = pd.read_csv('t1/preds/t1_pipeline/t1_blend.csv'); s = np.load('t1/research/blind_sonnet/stack_lr_test.npy')
assert (o.id == A.id).all() and (p.id == A.id).all() and len(s) == len(A)
base = np.load(Q + 'r13_test_base.npy')


def t1(comp, rate=.60):
    b = .7 * comp + .2 * R(s) + .1 * R(p.score.values); return np.where(b >= np.quantile(b, 1 - rate), '1.0', '0.0')


assert (t1(R(o.score.values)) == A.task1.values).all()   # reproduces v21a T1
print('corr base vs final_v2', round(np.corrcoef(R(base), R(o.score.values))[0, 1], 4))
for name, v in [('sub_v22a_LEAK_T1pseudoSoftW05.csv', 'soft_w05'), ('sub_v22b_LEAK_T1pseudoHard.csv', 'hard')]:
    d = A.copy(); d['task1'] = t1(R(o.score.values) + R(np.load(Q + f'r13_test_{v}.npy')) - R(base))
    d.to_csv(F + name, index=False, lineterminator='\n'); r = pd.read_csv(F + name, dtype=str, keep_default_na=False)
    assert r.shape == (25889, 6) and (r == '').sum().sum() == 0 and b'\r' not in open(F + name, 'rb').read() and list(r.columns) == list(A.columns)
    print(name, '| changed vs v21a', {c: int((A[c] != r[c]).sum()) for c in A.columns if (A[c] != r[c]).any()},
          '| T1 rate', round((r.task1 == '1.0').mean(), 4), '| id', r.id.iloc[0], r.id.iloc[-1])

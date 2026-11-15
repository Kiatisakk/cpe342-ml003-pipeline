# v24a: sub_v23a + T3 = 10-seed bag of the final T3 recipe (Kaggle kernel task3-seedbag10:
#   10 LightGBM p seeds x 10 CatBoost d4 MultiQuantile seeds, geo-mean Q, hurdle median).
# OOF 0.79537 vs 0.79529 (bag_q3p3 = current t3_best_test, reproduced on Kaggle to 3e-10). Everything else = v23a.
import numpy as np, pandas as pd
F = 'submissions/FINAL/'
A = pd.read_csv(F + 'sub_v23a_LEAK_T4blendEva02.csv', dtype=str, keep_default_na=False)
cur = pd.read_csv('t3/research/t3_best_test.csv')
assert np.allclose(A.task3.astype(float).values, cur.pred.values, rtol=0, atol=0.006), 'v23a T3 != t3_best_test'
t = np.clip(np.load('t3/preds/seedbag10_final/bag_q10p10_test.npy'), 0, None); assert len(t) == len(A) and np.isfinite(t).all()
d = A.copy(); d['task3'] = [str(round(float(v), 2)) for v in t]   # v23a writes T3 rounded to 2 decimals
name = 'sub_v24a_LEAK_T3seedbag10.csv'
d.to_csv(F + name, index=False, lineterminator='\n'); r = pd.read_csv(F + name, dtype=str, keep_default_na=False)
assert r.shape == (25889, 6) and (r == '').sum().sum() == 0 and b'\r' not in open(F + name, 'rb').read() and list(r.columns) == list(A.columns)
print(name, '| changed vs v23a', {c: int((A[c] != r[c]).sum()) for c in A.columns if (A[c] != r[c]).any()},
      '| T3 mean', round(t.mean(), 1), 'zero%', round((t == 0).mean(), 4), '| sample', A.task3.iloc[:3].tolist(), r.task3.iloc[:3].tolist())

# v24b (no leak): sub_v20a + T3 seed bag + T4 v3/EVA02 blend, both taken from v24a (neither uses player_id).
B = pd.read_csv(F + 'sub_v20a_noleak_T1blend721.csv', dtype=str, keep_default_na=False)
assert (B.id == r.id).all()
e = B.copy(); e['task3'] = r.task3; e['task4'] = r.task4
name = 'sub_v24b_noleak_T3seedbag10_T4blendEva02.csv'
e.to_csv(F + name, index=False, lineterminator='\n'); q = pd.read_csv(F + name, dtype=str, keep_default_na=False)
assert q.shape == (25889, 6) and (q == '').sum().sum() == 0 and b'\r' not in open(F + name, 'rb').read() and list(q.columns) == list(B.columns)
print(name, '| changed vs v20a', {c: int((B[c] != q[c]).sum()) for c in B.columns if (B[c] != q[c]).any()})

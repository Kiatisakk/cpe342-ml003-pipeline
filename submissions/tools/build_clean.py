# Strict-clean submission: v25a base (T2/T3/T5) + clean T1 (t1/preds/clean_fixedthr) + a T4 blend dir.
# usage: python submissions/tools/build_clean.py <t4 blend dir> <output name>
import sys, subprocess, os, pandas as pd
src, name = sys.argv[1], sys.argv[2]; F = 'submissions/FINAL/'
subprocess.run([sys.executable, 'submissions/tools/build_v26_t4.py', src, '_tmp'], check=True, stdout=subprocess.DEVNULL)
d = pd.read_csv(F + 'sub_v26__tmp.csv', dtype=str, keep_default_na=False); os.remove(F + 'sub_v26__tmp.csv')
c = pd.read_csv('t1/preds/clean_fixedthr/t1_clean.csv'); assert (c.id == d.id).all()
d['task1'] = c.pred.map({1: '1.0', 0: '0.0'}).values
O = F + name; d.to_csv(O, index=False, lineterminator='\n')
r = pd.read_csv(O, dtype=str, keep_default_na=False); assert r.shape == (25889, 6) and (r == '').sum().sum() == 0 and b'\r' not in open(O, 'rb').read()
for b in ['sub_v43_strictclean', 'sub_v37_T4frozen4_ft5L_emfull']:
    e = pd.read_csv(F + b + '.csv', dtype=str, keep_default_na=False); print(name, 'vs', b, {k: int((r[k] != e[k]).sum()) for k in r.columns[1:] if (r[k] != e[k]).any()})

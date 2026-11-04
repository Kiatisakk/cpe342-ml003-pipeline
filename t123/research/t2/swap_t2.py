"""Replace the task2 column of an existing 5-task submission with the player_id-block (leak) T2 predictions.
usage (Windows cmd, from repo root):
  python t123/research/t2/swap_t2.py submissions\\sub_v6_t1gbmblend_t2nb_t3blend_t4ens5f_ovr_t5union.csv submissions\\sub_v7_t2idleak.csv
Keeps every other column untouched, checks row count / id order, writes floats, comma, LF. Does NOT submit."""
import sys, pandas as pd
src, dst = sys.argv[1], sys.argv[2]
sub = pd.read_csv(src); t2 = pd.read_csv('t123/research/t2/t2_idleak_test.csv')
assert len(sub) == len(t2) == 25889 and (sub['id'].values == t2['id'].values).all(), 'row count / id order mismatch'
old = sub['task2'].round().astype(int).values
sub['task2'] = t2['pred'].astype(float)
print(f'task2 changed on {(old != t2.pred.values).mean():.1%} of rows; class share', (t2.pred.value_counts(normalize=True).sort_index().round(3)).to_dict())
sub.to_csv(dst, index=False, lineterminator='\n'); print('wrote', dst)

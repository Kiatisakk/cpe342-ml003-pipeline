# T1 "strict clean": same models and weights as sub_v20a, but no test-set statistics are used.
# Each model's test score is mapped to its position in that model's OOF distribution (ECDF of train OOF),
# instead of ranking test rows against each other; the cut is a fixed value learned on OOF.
import numpy as np, pandas as pd
from sklearn.metrics import fbeta_score
tr = pd.read_csv('cpe342-kaggle-main/01_binary_classification/train.csv'); lab = tr.is_cheater.notna().values; y = tr.is_cheater.values[lab].astype(int)
def ecdf(ref):                                   # score -> fraction of OOF scores below it
    s = np.sort(ref); return lambda v: np.searchsorted(s, v, side='right') / len(s)
def pair(oof, te): f = ecdf(oof); return f(oof), f(te)
# final_v2 (3-seed mean), blind stack_lr, GBM pipeline (xgb .1834 / lgb .4848 / cat .3318 on ranks)
o_oof, o_te = pair(np.load('t1/research/best_oof_allseeds.npy').mean(0), np.load('t1/research/best_test_proba.npy'))
s_all = np.load('t1/research/blind_sonnet/stack_lr_oof.npy'); s_oof, s_te = pair(s_all[lab] if len(s_all) == len(tr) else s_all, np.load('t1/research/blind_sonnet/stack_lr_test.npy'))
W = {'xgb': .1834, 'lgb': .4848, 'cat': .3318}; P = 't1/preds/t1_pipeline/'
g = [pair(np.load(f'{P}oof_{m}.npy'), np.load(f'{P}test_{m}.npy')) for m in W]
g_oof = sum(w * a for w, (a, _) in zip(W.values(), g)); g_te = sum(w * b for w, (_, b) in zip(W.values(), g))
p_oof, p_te = pair(g_oof, g_te)
b_oof = .7 * o_oof + .2 * s_oof + .1 * p_oof; b_te = .7 * o_te + .2 * s_te + .1 * p_te
assert len(b_oof) == len(y)
thr60 = np.quantile(b_oof, 0.40)                 # v20a's 60% rate, for comparison only
grid = np.quantile(b_oof, np.linspace(.2, .7, 251)); f2 = [fbeta_score(y, b_oof >= t, beta=2) for t in grid]; tb = grid[int(np.argmax(f2))]
print('OOF binary F2: thr60', round(fbeta_score(y, b_oof >= thr60, beta=2), 5), '| best', round(max(f2), 5), 'at OOF pos rate', round((b_oof >= tb).mean(), 3))
pred = (b_te >= tb).astype(int)                 # cut = OOF binary-F2 optimum (train only)
ids = pd.read_csv('t1/research/t1_best_test.csv').id
pd.DataFrame({'id': ids, 'pred': pred, 'score': b_te}).to_csv('t1/preds/clean_fixedthr/t1_clean.csv', index=False, lineterminator='\n')
print('test positive rate', round(pred.mean(), 4))

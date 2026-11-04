"""Write the deliverables.
Feature-only (rule-safe, no player_id):  best_oof_proba.npy, best_test_proba.npy, t2_best_test.csv
   = label-noise-aware EM Naive Bayes (e06, per-feature Optuna histogram cfg of the baseline).
     OOF = seed-42 5-fold OOF probabilities (P(observed label | x)); test = model refit on ALL train rows.
ID-leak variant (see REPORT.md, team decision): idleak_oof_proba.npy, idleak_test_proba.npy, t2_idleak_test.csv
   = player_id block rule with NB tie-break inside the (tiny) boundary gaps, cut points fitted on all train rows.
usage: python t123/research/t2/make_final.py"""
import sys; sys.argv = sys.argv[:1] + ['15', 'opt', 'em', '1.0']
sys.path.insert(0, 't123/research/t2')
from runner import *
import e06_emnb as E
import e10_idblock as I

ids = [f'ANS{i:05d}' for i in range(1, NT + 1)]

# ---- feature-only best: EM-NB refit on all train
a = np.arange(N); B = E.binned(a, E.CFG); tabs, pi, eps = E.fit(a, B)
M = np.full((K, K), eps / 3); np.fill_diagonal(M, 1 - eps)
Pte = softmax(E.loglik(B, tabs, test=True) + np.log(pi)) @ M
oof = load('emnb_em_opt_it15_a1')[0][0]
print('EM-NB eps(full train) =', round(eps, 4), '| seed-42 OOF F1', round(f1m(y, oof), 4))
np.save(f'{OUT}/best_oof_proba.npy', oof.astype(np.float32)); np.save(f'{OUT}/best_test_proba.npy', Pte.astype(np.float32))
pd.DataFrame({'id': ids, 'pred': Pte.argmax(1)}).to_csv(f'{OUT}/t2_best_test.csv', index=False, lineterminator='\n')

# agreement with the existing submission file
old = pd.read_csv('t123/preds_final/t2_nb.csv')
print('agreement with t123/preds_final/t2_nb.csv:', round((old.pred.values == Pte.argmax(1)).mean(), 4))
print('test class share:', np.bincount(Pte.argmax(1), minlength=K) / NT)

# ---- ID-leak variant
S_te_full = E.loglik(B, tabs, test=True) + np.log(pi)          # NB scores for gap tie-breaks
cs, gaps = I.cuts(a)
blk = np.digitize(I.IDT, cs); P = np.full((NT, K), 0.01); P[np.arange(NT), blk] = 0.97
ng = 0
for k, (lo, hi) in enumerate(gaps):
    g = (I.IDT > lo) & (I.IDT < hi); ng += g.sum()
    if g.any():
        s2 = softmax(S_te_full[g][:, [k, k + 1]]); P[g] = 0.01; P[g, k] = 0.01 + 0.96 * s2[:, 0]; P[g, k + 1] = 0.01 + 0.96 * s2[:, 1]
print('ID cuts', cs, 'test rows inside gaps:', ng)
np.save(f'{OUT}/idleak_oof_proba.npy', load('idblock')[0][0]); np.save(f'{OUT}/idleak_test_proba.npy', P.astype(np.float32))
pd.DataFrame({'id': ids, 'pred': P.argmax(1)}).to_csv(f'{OUT}/t2_idleak_test.csv', index=False, lineterminator='\n')
print('ID-leak vs feature-only test agreement:', round((P.argmax(1) == Pte.argmax(1)).mean(), 4),
      '(expected ~ NB accuracy vs true class 0.79)')

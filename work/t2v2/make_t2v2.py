"""T2 on the re-released data (12 Nov): same recipe as sub_v18a/v20a T2, no player_id.
EM-NB with LL-selected densities refit on all train + blend .87 NB / .08 spline LR / .05 MLP + fixed-point affine F1 rule.
Run from repo root after e14_llselect.py (seeds 42/7/123, g2, em 1) and blind fam.py lin / nn.py in work/t2v2/blind."""
import sys; sys.argv = sys.argv[:1] + ['15', 'opt', 'em', '1.0']
sys.path.insert(0, 't2/research')
from runner import *
import e06_emnb as E
OUTD = 't2/preds/v2data'; os.makedirs(OUTD, exist_ok=True)
a = np.arange(N)
GNB, GSM = [64, 128, 256], [3.0, 4.0, 6.0, 8.0, 12.0]
inner = list(StratifiedKFold(4, shuffle=True, random_state=0).split(a, y))
cfg = {}
for c in CONT:
    best = None
    for nb_ in GNB:
        for sm in GSM:
            ll = 0
            for ia, ib in inner:
                L = loglik_block(c, ia, nb_, sm, 1.0)[0][ib] + logprior(ia)
                ll += np.mean(np.log(softmax(L)[np.arange(len(ib)), y[ib]]))
            if best is None or ll > best[0]: best = (ll, nb_, sm)
    cfg[c] = best[1:]
B = E.binned(a, cfg); tabs, pi_, eps = E.fit(a, B)
M = np.full((K, K), eps / 3); np.fill_diagonal(M, 1 - eps)
Pte = softmax(E.loglik(B, tabs, test=True) + np.log(pi_)) @ M
oof = load('nb_llsel_g2_em1')[0][0]
np.save(f'{OUTD}/nb_oof.npy', oof.astype(np.float32)); np.save(f'{OUTD}/nb_test.npy', Pte.astype(np.float32))
S = 'work/t2v2/blind/out/'
Bo = .87 * oof + .08 * np.load(S + 'logreg_spline_A_oof.npy') + .05 * np.load(S + 'mlp_torch_A_oof.npy')
Bt = .87 * Pte + .08 * np.load(S + 'logreg_spline_A_test.npy') + .05 * np.load(S + 'mlp_torch_A_test.npy')
pi = np.bincount(y, minlength=4) / N; q = pi.copy(); Fk = np.full(4, .75)
for _ in range(8):
    aa = 2 / (pi + q); bb = -Fk / (pi + q); pr = (aa * Bo + bb).argmax(1); q = np.bincount(pr, minlength=4) / N
    Fk = np.array([2 * ((pr == k) & (y == k)).mean() / (pi[k] + q[k]) for k in range(4)])
print('eps', round(eps, 4), '| OOF F1m NB', round(f1m(y, oof), 5), 'blend', round(f1m(y, Bo), 5), 'blend+affine', round(f1_score(y, pr, average='macro'), 5))
pred = (aa * Bt + bb).argmax(1)
ids = [f'ANS{i:05d}' for i in range(1, NT + 1)]
pd.DataFrame({'id': ids, 'pred': pred}).to_csv(f'{OUTD}/t2v2_blend_affine.csv', index=False, lineterminator='\n')
pd.DataFrame({'id': ids, 'pred': Pte.argmax(1)}).to_csv(f'{OUTD}/t2v2_nb.csv', index=False, lineterminator='\n')
print('test share', np.round(np.bincount(pred, minlength=4) / NT, 3), 'train prior', np.round(pi, 3))

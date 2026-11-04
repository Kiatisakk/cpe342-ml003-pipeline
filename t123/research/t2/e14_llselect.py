"""E14: honest per-feature density selection by inner-CV UNIVARIATE POSTERIOR log-loss.
pf.py's 'll_select' compared histogram log-MASS across different bin counts, which always favours coarse bins (bug).
Here, for each continuous feature, (nbins, sigma) minimises -mean log softmax(log p(x_j|k) + log prior)[y] over a
4-fold inner CV of the outer-train rows only -> refit on outer-train -> score the outer fold. Nested by construction.
usage: python t123/research/t2/e14_llselect.py <seed> [grid: g1|g2] [em 0/1]
  g1 = original grid, g2 = extended (finer bins, wider smoothing; g1 picks hit the 128-bin / sigma-4 edge), g3 = further still"""
import sys, json; sys.path.insert(0, 't123/research/t2')
from common import *
S_ = int(sys.argv[1]); GR = sys.argv[2] if len(sys.argv) > 2 else 'g1'; EM = int(sys.argv[3]) if len(sys.argv) > 3 else 0
NB, SM = ([8, 16, 24, 32, 48, 64, 96, 128], [0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]) if GR == 'g1' else ([64, 128, 256], [3.0, 4.0, 6.0, 8.0, 12.0]) if GR == 'g2' else ([128, 256, 512], [6.0, 8.0, 12.0, 16.0, 24.0])
TAG = 'nb_llsel' if (GR, EM) == ('g1', 0) else f'nb_llsel_{GR}_em{EM}'
if EM:
    sys.argv = sys.argv[:1] + ['15', 'opt', 'em', '1.0']; import e06_emnb as E
oof = np.zeros((N, K)); te_ = np.zeros((NT, K)); chosen = []
for fo, (A, Bo) in enumerate(folds(S_)):
    inner = list(StratifiedKFold(4, shuffle=True, random_state=fo).split(A, y[A]))
    cfg = {}
    for c in CONT:
        best = None
        for nb_ in NB:
            for sm in SM:
                ll = 0
                for ia, ib in inner:
                    L = loglik_block(c, A[ia], nb_, sm, 1.0)[0][A[ib]] + logprior(A[ia])
                    ll += np.mean(np.log(softmax(L)[np.arange(len(ib)), y[A[ib]]]))
                if best is None or ll > best[0]: best = (ll, nb_, sm)
        cfg[c] = best[1:]
    chosen.append(cfg)
    if EM:
        Bn = E.binned(A, cfg); tabs, pi, eps = E.fit(A, Bn)
        M = np.full((K, K), eps / 3); np.fill_diagonal(M, 1 - eps)
        St = np.zeros((N, K)); St[Bo] = E.loglik(Bn, tabs, Bo) + np.log(pi); Se = E.loglik(Bn, tabs, test=True) + np.log(pi)
        oof[Bo] = softmax(St[Bo]) @ M; te_ += softmax(Se) @ M / 5
    else:
        St = logprior(A) + np.zeros((N, K)); Se = logprior(A) + np.zeros((NT, K))
        for c in FEATS:
            Lt, Le = loglik_block(c, A, *(cfg[c] if c in CONT else (0, 0)), 1.0); St += Lt; Se += Le
        oof[Bo] = softmax(St[Bo]); te_ += softmax(Se) / 5
    print(S_, fo, round(f1_score(y[Bo], St[Bo].argmax(1), average='macro'), 4), flush=True)
np.savez_compressed(f'{CACHE}/P_{TAG}_s{S_}.npz', oof=oof.astype(np.float32), te=te_)
json.dump([{c: list(v) for c, v in cf.items()} for cf in chosen], open(f'{CACHE}/{TAG}_cfg_s{S_}.json', 'w'))
print('seed', S_, 'F1', round(f1m(y, oof), 4), 'cfg fold0', chosen[0], flush=True)

"""E16: ensemble weight blending (features only, NO player_id), weights learned NESTED.
For each seed and each outer fold f: weights are fitted on the OOF rows of the other 4 folds, applied to fold f.
Members exclude models built on the baseline's overfit Optuna cfg (nb_opt, emnb_opt, nbbag; see E13).
Blends:
  best_single : nb_llsel_g2_em1
  equal_geo   : equal-weight geometric mean (no fitting)
  lin_ll      : linear pool  sum_m w_m P_m,              w on simplex, fitted by log-loss
  geo_ll      : log-linear pool softmax(sum_m w_m log P_m), w >= 0, fitted by log-loss
  geo_f1      : log-linear pool, w fitted by macro-F1 (random simplex search + local refinement)
each also with a nested class-bias. Robustness = sd over seeds and over the 15 (seed, fold) scores.
usage: python t2/research/e16_weight_blend.py m1 m2 ...   (default member list below)"""
import sys, json; sys.path.insert(0, 't2/research')
from runner import *
from scipy.optimize import minimize
MEM = sys.argv[1:] or ['nb_llsel_g2_em1', 'nb_llsel_g2_em0', 'emnb_em_base_it15_a1', 'gam_k8', 'gamnc_k8_l0.0001_nc1', 'lgbm_raw']
D = {m: load(m) for m in MEM}
OOF = np.stack([np.log(np.clip(D[m][0], 1e-12, 1)) for m in MEM], 1)   # (seeds, M, N, K) log-probas
TE = np.stack([np.log(np.clip(D[m][1], 1e-12, 1)) for m in MEM], 0)    # (M, NT, K)
M_ = len(MEM)


def pool(LP, w, kind):
    if kind == 'lin': return np.log(np.einsum('m,mnk->nk', w, np.exp(LP)) + 1e-12)
    return np.log(softmax(np.einsum('m,mnk->nk', w, LP)) + 1e-12)


def fit_ll(LP, yy, kind):
    def f(p):
        w = np.exp(p) / np.exp(p).sum() if kind == 'lin' else np.exp(p)
        S = pool(LP, w, kind); return -S[np.arange(len(yy)), yy].mean()
    r = minimize(f, np.zeros(M_) if kind == 'lin' else np.log(np.full(M_, 1 / M_)), method='Nelder-Mead',
                 options=dict(maxiter=400 * M_, xatol=1e-4, fatol=1e-7))
    return np.exp(r.x) / np.exp(r.x).sum() if kind == 'lin' else np.exp(r.x)


def fit_f1(LP, yy, rng):
    best = (-1, None)
    cands = [np.full(M_, 1 / M_)] + [rng.dirichlet(np.ones(M_)) for _ in range(150)]
    for w in cands:
        v = _f1_fast(yy, pool(LP, w, 'geo').argmax(1))
        if v > best[0]: best = (v, w)
    w = best[1]
    for _ in range(60):                                   # local refinement
        w2 = np.clip(w + rng.normal(0, 0.05, M_), 0, None); w2 /= w2.sum()
        v = _f1_fast(yy, pool(LP, w2, 'geo').argmax(1))
        if v > best[0]: best, w = (v, w2), w2
    return w


from common import _f1_fast
methods = ['best_single', 'equal_geo', 'lin_ll', 'geo_ll', 'geo_f1']
res = {f'{m}{b}': [] for m in methods for b in ('', '+bias')}; perfold = {k: [] for k in res}; W = {m: [] for m in methods[2:]}
for si, s in enumerate(SEEDS):
    LP = OOF[si]; preds = {k: np.zeros(N, int) for k in res}; rng = np.random.RandomState(s)
    for a, b in folds(s):
        for m in methods:
            if m == 'best_single': Sa, Sb = LP[0][a], LP[0][b]
            elif m == 'equal_geo': w = np.full(M_, 1 / M_); Sa, Sb = pool(LP[:, a], w, 'geo'), pool(LP[:, b], w, 'geo')
            else:
                kind = 'lin' if m == 'lin_ll' else 'geo'
                w = fit_ll(LP[:, a], y[a], kind) if m != 'geo_f1' else fit_f1(LP[:, a], y[a], rng)
                W[m].append(w); Sa, Sb = pool(LP[:, a], w, kind), pool(LP[:, b], w, kind)
            preds[m][b] = Sb.argmax(1)
            preds[m + '+bias'][b] = (Sb + tune_bias(Sa, y[a])).argmax(1)
            for k in (m, m + '+bias'): perfold[k].append(f1_score(y[b], preds[k][b], average='macro'))
    for k in res: res[k].append(f1_score(y, preds[k], average='macro'))
print('members:', MEM)
print(f'{"blend":18s} {"macro-F1 per seed":26s} mean     sd(seeds) sd(15 folds)')
for k, v in res.items():
    print(f'{k:18s} {" ".join(f"{x:.4f}" for x in v):26s} {np.mean(v):.4f}   {np.std(v):.4f}    {np.std(perfold[k]):.4f}')
for m, ws in W.items():
    print(f'mean weights {m:7s}', dict(zip(MEM, np.round(np.mean(ws, 0), 3))), ' sd', np.round(np.std(ws, 0), 3))
json.dump({'members': MEM, 'res': res, 'weights': {m: np.mean(ws, 0).tolist() for m, ws in W.items()}},
          open(f'{CACHE}/e16_res.json', 'w'), indent=1)

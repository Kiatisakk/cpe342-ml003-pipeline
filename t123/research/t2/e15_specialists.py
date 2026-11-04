"""E15: class-specialist (one-vs-rest) models for T2, features only (no player_id).
Per outer fold (densities / models fitted on outer-train rows only):
  P_nb  : 4-class NB (uniform cfg, alpha 1)
  s_k   : binary LightGBM "class k vs rest", inputs = raw features + honest inner-OOF NB log-posteriors (4)
Decision rules evaluated (all thresholds / biases nested on the other folds' OOF rows):
  nb          : argmax P_nb
  hier0       : predict 0 if s_0 >= t, else argmax over classes 1..3 of P_nb      (the "train only 0" idea)
  swap0       : P_nb with column 0 replaced by s_0 (renormalised), argmax + nested bias
  ovr         : argmax_k s_k + nested bias                                         (4 specialists)
  ovr+nb      : geometric mean of normalised specialists and P_nb, + nested bias
usage: python t123/research/t2/e15_specialists.py <seed>"""
import sys; sys.path.insert(0, 't123/research/t2')
import json
from runner import *
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from e05_lgbm import raw
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 42
TH = int(os.environ.get('OMP_NUM_THREADS', 4))
P = dict(objective='binary', learning_rate=0.03, num_leaves=15, min_child_samples=100, feature_fraction=0.7,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, num_threads=TH, seed=SEED)
FS = folds(SEED)
Pnb = np.zeros((N, K)); Sp = np.zeros((N, K))
for fo, (a, b) in enumerate(FS):
    Lin, Lall, Lte = inner_blocks(a, CFG_BASE, seed=fo)
    Pnb[b] = softmax(Lall[b].sum(1) + logprior(a))
    nbf_a = np.log(softmax(Lin.sum(1) + logprior(a)) + 1e-12); nbf_b = np.log(Pnb[b] + 1e-12)
    Xt, _ = raw(a); Xa = np.c_[Xt[a], nbf_a]; Xb = np.c_[Xt[b], nbf_b]
    ia, iv = train_test_split(np.arange(len(a)), test_size=0.1, stratify=y[a], random_state=0)
    for k in range(K):
        yk = (y[a] == k).astype(int)
        m = lgb.train(P, lgb.Dataset(Xa[ia], yk[ia]), 5000, valid_sets=[lgb.Dataset(Xa[iv], yk[iv])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        Sp[b, k] = m.predict(Xb)
    print(f'seed {SEED} fold {fo}: NB {f1m(y[b], Pnb[b]):.4f}  OvR {f1m(y[b], Sp[b]):.4f}', flush=True)
np.savez_compressed(f'{CACHE}/e15_s{SEED}.npz', Pnb=Pnb, Sp=Sp)


def nested(rule):
    pred = np.zeros(N, int)
    for a, b in FS: pred[b] = rule(a, b)
    return f1_score(y, pred, average='macro')


def hier0(a, b):
    def pr(idx, t): return np.where(Sp[idx, 0] >= t, 0, 1 + Pnb[idx, 1:].argmax(1))
    grid = np.unique(np.quantile(Sp[a, 0], np.linspace(0.3, 0.8, 101)))
    t = max(grid, key=lambda g: f1_score(y[a], pr(a, g), average='macro'))
    return pr(b, t)


def biased(S):
    return lambda a, b: (S[b] + tune_bias(S[a], y[a])).argmax(1)


L = lambda M: np.log(np.clip(M, 1e-12, 1))
sw = Pnb.copy(); sw[:, 0] = Sp[:, 0]
sw[:, 1:] = Pnb[:, 1:] / Pnb[:, 1:].sum(1, keepdims=True) * (1 - Sp[:, [0]])
Sn = Sp / Sp.sum(1, keepdims=True)
res = {'nb': f1m(y, Pnb), 'nb+bias': nested(biased(L(Pnb))), 'hier0': nested(hier0),
       'swap0+bias': nested(biased(L(sw))), 'ovr+bias': nested(biased(L(Sn))),
       'ovr+nb+bias': nested(biased((L(Sn) + L(Pnb)) / 2))}
for k in range(K):
    from sklearn.metrics import roc_auc_score
    print(f'class {k}: AUC one-vs-rest  NB {roc_auc_score(y == k, Pnb[:, k]):.4f}  specialist {roc_auc_score(y == k, Sp[:, k]):.4f}')
for n, v in res.items(): print(f'{n:14s} macro-F1 {v:.4f}')
json.dump(res, open(f'{CACHE}/e15_res_s{SEED}.json', 'w'), indent=1)

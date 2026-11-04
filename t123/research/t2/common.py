"""Shared helpers for T2 research. Run scripts from the repo root.

Reproduces the baseline histogram Naive Bayes of t123/t2_nb/nb.py exactly, but
generalised so that densities fitted on a set of TRAIN rows can be evaluated on
any rows (OOF rows or test rows). Test rows are never used for fitting anything:
bin edges, level maps, counts all come from training rows only.
"""
import os, json, numpy as np, pandas as pd, warnings
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
warnings.filterwarnings('ignore')

ROOT = 'cpe342-kaggle-main/02_multiclass_classification'
OUT = 't123/research/t2'
CACHE = os.path.join(OUT, 'cache'); os.makedirs(CACHE, exist_ok=True)
K = 4
SEEDS = (42, 7, 123)

tr = pd.read_csv(f'{ROOT}/train.csv')
te = pd.read_csv(f'{ROOT}/test.csv')
y = tr.segment.astype(int).values.copy()
N, NT = len(tr), len(te)

NOISE = ['random_metric_1', 'random_metric_2', 'random_metric_3', 'account_age_days', 'responds_to_discounts',
         'item_type_preference_social', 'avg_match_length', 'preferred_game_mode', 'region', 'platform', 'device_type',
         'payment_method', 'language', 'account_status', 'player_type_tag', 'skill_tier']
FEATS = [c for c in tr.columns if c not in ['player_id', 'segment'] + NOISE]
DISC = [c for c in FEATS if (not pd.api.types.is_numeric_dtype(tr[c])) or tr[c].nunique() <= 12]
CONT = [c for c in FEATS if c not in DISC]

BEST = json.load(open('t123/t2_nb/best_nb_perfeature.json'))
CFG_OPT = {c: tuple(v) for c, v in BEST['optuna']['cfg'].items()}   # per-feature (nbins, sigma), alpha=1
CFG_BASE = {c: (32, 0.7) for c in CONT}


def folds(seed):
    return list(StratifiedKFold(5, shuffle=True, random_state=seed).split(tr, y))


def f1m(yt, P):
    return f1_score(yt, np.asarray(P).argmax(1), average='macro')


def _disc_codes(c, a):
    """map strings to codes using levels seen in train rows a; unseen levels in other rows -> -1 (ignored)."""
    vt = tr[c].astype(object).where(tr[c].notna(), 'NA').astype(str).values
    ve = te[c].astype(object).where(te[c].notna(), 'NA').astype(str).values
    lv = sorted(set(vt[a]))
    m = {x: i for i, x in enumerate(lv)}
    return np.array([m.get(x, -1) for x in vt]), np.array([m.get(x, -1) for x in ve]), len(lv)


def loglik_block(c, a, nbins=32, sm=0.7, alpha=1.0):
    """per-class log p(x_c|k), densities from train rows a.
    returns (N,K) for all train rows and (NT,K) for test rows. Identical to nb.loglik for rows in b."""
    if c in DISC:
        bt, be, n = _disc_codes(c, a)
        nant, nane = bt < 0, be < 0
        bt, be = np.maximum(bt, 0), np.maximum(be, 0)
        keepmask = np.ones(N, bool)
    else:
        x = tr[c].values.astype(float); xe = te[c].values.astype(float)
        e = np.unique(np.nanquantile(x[a], np.linspace(0, 1, nbins + 1))); n = len(e) - 1
        nant, nane = np.isnan(x), np.isnan(xe)
        bt = np.where(nant, 0, np.clip(np.searchsorted(e, x, side='right') - 1, 0, n - 1))
        be = np.where(nane, 0, np.clip(np.searchsorted(e, np.nan_to_num(xe), side='right') - 1, 0, n - 1))
        keepmask = ~nant
    Lt = np.zeros((N, K)); Le = np.zeros((NT, K))
    for k in range(K):
        sel = a[(y[a] == k) & keepmask[a]]
        cnt = np.bincount(bt[sel], minlength=n).astype(float)
        if sm > 0 and c in CONT:
            cnt = gaussian_filter1d(cnt, sm, mode='nearest')
        lp = np.log((cnt + alpha) / (cnt + alpha).sum())
        Lt[:, k] = lp[bt]; Le[:, k] = lp[be]
    Lt[nant] = 0; Le[nane] = 0
    return Lt, Le


def nb_blocks(cfg=CFG_OPT, alpha=1.0, seed=42, tag='opt'):
    """OOF per-feature log-lik blocks (N,F,K) for 5 folds of `seed` + fold-model test blocks averaged (NT,F,K)
    + full-train test blocks (NT,F,K). Cached."""
    fn = os.path.join(CACHE, f'nbblk_{tag}_{seed}.npz')
    if os.path.exists(fn):
        z = np.load(fn); return z['oof'], z['te']
    F = len(FEATS)
    oof = np.zeros((N, F, K), np.float32); tst = np.zeros((NT, F, K), np.float32)
    for a, b in folds(seed):
        for j, c in enumerate(FEATS):
            nb_, sm = cfg.get(c, (0, 0)) if c in CONT else (0, 0)
            Lt, Le = loglik_block(c, a, nb_, sm, alpha)
            oof[b, j] = Lt[b]; tst[:, j] += Le / 5
    np.savez_compressed(fn, oof=oof, te=tst)
    return oof, tst


def nb_full_test(cfg=CFG_OPT, alpha=1.0):
    a = np.arange(N); F = len(FEATS)
    full_tr = np.zeros((N, F, K), np.float32); tst = np.zeros((NT, F, K), np.float32)
    for j, c in enumerate(FEATS):
        nb_, sm = cfg.get(c, (0, 0)) if c in CONT else (0, 0)
        full_tr[:, j], tst[:, j] = loglik_block(c, a, nb_, sm, alpha)
    return full_tr, tst


def logprior(idx):
    return np.log(np.bincount(y[idx], minlength=K) / len(idx))


def softmax(S):
    S = S - S.max(1, keepdims=True); P = np.exp(S); return P / P.sum(1, keepdims=True)


def report(name, scores):
    s = np.array(scores)
    print(f'{name:45s} ' + ' '.join(f'{v:.4f}' for v in s) + f' | mean {s.mean():.4f} sd {s.std():.4f}', flush=True)
    return s.mean(), s.std()


def _f1_fast(yt, pred):
    cm = np.bincount(yt * K + pred, minlength=K * K).reshape(K, K)
    tp = np.diag(cm); return np.mean(2 * tp / np.maximum(cm.sum(0) + cm.sum(1), 1))


def tune_bias(S, yt, sweeps=3, grid=np.arange(-0.6, 0.601, 0.02)):
    """coordinate search for an additive class-bias vector (class 0 fixed at 0) maximising macro-F1 of argmax(S+b)."""
    b = np.zeros(K)
    best = _f1_fast(yt, (S + b).argmax(1))
    for _ in range(sweeps):
        for k in range(1, K):
            for g in grid:
                bb = b.copy(); bb[k] = g
                v = _f1_fast(yt, (S + bb).argmax(1))
                if v > best + 1e-9: best, b = v, bb
    return b


def nested_bias_f1(S, seed):
    """S: OOF log-scores (N,K) for folds(seed). For each outer fold tune bias on the other 4 folds' OOF rows,
    apply to the held-out fold. Returns macro-F1 of the nested predictions and the per-fold biases."""
    pred = np.zeros(N, int); bs = []
    for a, b in folds(seed):
        bb = tune_bias(S[a], y[a]); bs.append(bb); pred[b] = (S[b] + bb).argmax(1)
    return f1_score(y, pred, average='macro'), np.array(bs)


def inner_blocks(a, cfg=CFG_OPT, alpha=1.0, seed=0):
    """Honest stacking features for an outer training set a:
    inner-OOF NB blocks for rows a (5-fold inside a), and blocks for ALL train rows + test fitted on all of a.
    returns (Lin (len(a),F,K), Lall (N,F,K), Lte (NT,F,K))"""
    F = len(FEATS)
    Lin = np.zeros((len(a), F, K), np.float32); Lall = np.zeros((N, F, K), np.float32); Lte = np.zeros((NT, F, K), np.float32)
    for ia, ib in StratifiedKFold(5, shuffle=True, random_state=seed).split(a, y[a]):
        for j, c in enumerate(FEATS):
            nb_, sm = cfg.get(c, (0, 0)) if c in CONT else (0, 0)
            Lt, _ = loglik_block(c, a[ia], nb_, sm, alpha); Lin[ib, j] = Lt[a[ib]]
    for j, c in enumerate(FEATS):
        nb_, sm = cfg.get(c, (0, 0)) if c in CONT else (0, 0)
        Lall[:, j], Lte[:, j] = loglik_block(c, a, nb_, sm, alpha)
    return Lin, Lall, Lte

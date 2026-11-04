"""Semi-supervised GBM: in-fold pseudo-labels on the 13,184 UNLABELED TRAIN rows (test is never touched).
Per outer fold: teacher LGB trained on the fold's training rows -> soft labels for unlabeled rows -> student
trained on fold-train + confident unlabeled rows (hard labels, weight w). Evaluated on the held-out fold."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import *
import lightgbm as lgb
lab, unl, te, ids = load(); y = lab[TARGET].astype(int).values
ref = pd.concat([lab, unl]); X = fe_v2(lab, ref); Xu = fe_v2(unl, ref)
CONF = float(os.environ.get("CONF", 0.15)); W = float(os.environ.get("PW", 0.5))


def fit(Xa, ya, Xv, yv, rs, w=None):
    m = lgb.LGBMClassifier(n_estimators=5000, learning_rate=0.05, random_state=rs, **LGB_BASE)
    m.fit(Xa, ya, sample_weight=w, eval_set=[(Xv, yv)], eval_metric="auc", callbacks=[lgb.early_stopping(200, verbose=False)]); return m


def run(seed):
    oof = np.zeros(len(y))
    for k, (a, b) in enumerate(folds_for(y, seed)):
        rng = np.random.default_rng(seed + k); a = rng.permutation(a); cut = int(0.9 * len(a)); tr, va = a[:cut], a[cut:]
        t = fit(X.iloc[tr], y[tr], X.iloc[va], y[va], seed + k)
        pu = t.predict_proba(Xu)[:, 1]; keep = (pu < CONF) | (pu > 1 - CONF)
        Xs = pd.concat([X.iloc[tr], Xu[keep]]); ys = np.concatenate([y[tr], (pu[keep] > 0.5).astype(int)])
        ws = np.concatenate([np.ones(len(tr)), np.full(keep.sum(), W)])
        oof[b] = fit(Xs, ys, X.iloc[va], y[va], seed + k, ws).predict_proba(X.iloc[b])[:, 1]
    return oof


a, o = multi_seed(f"exp11 lgb fe_v2 + pseudo-labelled unlabeled-train (conf {CONF}, w {W})", y, run)

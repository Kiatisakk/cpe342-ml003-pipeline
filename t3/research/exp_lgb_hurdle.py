# LightGBM hurdle: P(y>0) classifier + per-alpha quantile regressors for the positive part,
# predict the mixture median (q = 1 - 1/(2p), 0 if p < 0.5).
# usage: python t3/research/exp_lgb_hurdle.py NAME [--weight none|scale] [--target log|raw] [--seed 0] [--whale]
import sys, os, time, argparse, numpy as np, pandas as pd, lightgbm as lgb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load, nmae, fold_scores, save_oof, hurdle_median, extra_feats, prev_offset, NT
from sklearn.model_selection import train_test_split

ap = argparse.ArgumentParser()
ap.add_argument("name"); ap.add_argument("--weight", default="none"); ap.add_argument("--target", default="log")
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--lr", type=float, default=0.03)
ap.add_argument("--leaves", type=int, default=63); ap.add_argument("--mcs", type=int, default=40)
ap.add_argument("--offset", default="none"); ap.add_argument("--wpow", type=float, default=1.0); ap.add_argument("--clf_only", action="store_true")
ap.add_argument("--alphas", default="0.02,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5")
ap.add_argument("--frac", type=float, default=1.0)
ap.add_argument("--skip_clf", action="store_true", help="reuse saved 3-seed p")
a = ap.parse_args()
ALPHAS = np.array([float(x) for x in a.alphas.split(",")])
D = load(); X, Xt, y, F = D["X"], D["Xt"], D["y"], D["folds"]
if a.offset == "prev": X, Xt = extra_feats(X, Xt)
off = (lambda Z: prev_offset(Z)) if a.offset == "prev" else (lambda Z: np.zeros(len(Z)))
n, nt = len(y), len(Xt)

def P(obj, **kw):
    return {**dict(objective=obj, learning_rate=a.lr, num_leaves=a.leaves, min_child_samples=a.mcs, subsample=0.8,
                   subsample_freq=1, colsample_bytree=0.7, reg_lambda=2.0, n_estimators=10000, verbose=-1,
                   n_jobs=NT, random_state=a.seed), **kw}

def fit(model, Xa, ya, w=None, strat=None):
    i, j = train_test_split(np.arange(len(ya)), test_size=0.1, random_state=a.seed, stratify=strat)
    model.fit(Xa.iloc[i], ya[i], sample_weight=None if w is None else w[i], eval_set=[(Xa.iloc[j], ya[j])],
              eval_sample_weight=None if w is None else [w[j]], callbacks=[lgb.early_stopping(100, verbose=False)])
    return model

def scale_w(Xa):
    # feature-only weight ~ expected baht scale of the row (does NOT bias the conditional quantile)
    s = np.expm1(Xa["log_prev_month_spending"].fillna(Xa["log_historical_spending"] - 1.0).fillna(6.0).values)
    w = np.clip(s, 1, None) ** a.wpow
    return w / w.mean()

Op = np.zeros(n); Tp = np.zeros(nt); OQ = np.zeros((n, len(ALPHAS))); TQ = np.zeros((nt, len(ALPHAS)))
t0 = time.time()
for k, (tr, va) in enumerate(F):
    if a.frac < 1: tr = tr[np.random.RandomState(k).rand(len(tr)) < a.frac]
    Xa, ya = X.iloc[tr], y[tr]; pa = (ya > 0).astype(int)
    if not a.skip_clf:
        c = fit(lgb.LGBMClassifier(**P("binary")), Xa, pa, strat=pa)
        Op[va] = c.predict_proba(X.iloc[va])[:, 1]; Tp += c.predict_proba(Xt)[:, 1] / 5
    if a.clf_only: continue
    Xp, yp = Xa[pa == 1], ya[pa == 1]
    t = np.log(yp) - off(Xp) if a.target == "log" else yp
    w = scale_w(Xp) if a.weight == "scale" else None
    for j, al in enumerate(ALPHAS):
        m = fit(lgb.LGBMRegressor(**P("quantile", alpha=al)), Xp, t, w=w)
        ov, tv = m.predict(X.iloc[va]), m.predict(Xt)
        if a.target == "log": ov, tv = np.exp(ov + off(X.iloc[va])), np.exp(tv + off(Xt))
        OQ[va, j] = ov; TQ[:, j] += tv / 5
    print(f"fold {k} {time.time()-t0:.0f}s", flush=True)
if a.skip_clf:
    PP = np.mean([np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "oof", f"{q}_p.npy")) for q in ("lgbh_log", "clf_s1", "clf_s2")], 0); Op, Tp = PP[:n], PP[n:]
from sklearn.metrics import roc_auc_score
print("clf AUC", roc_auc_score(y > 0, Op))
np.save(os.path.join(os.path.dirname(__file__), "oof", f"{a.name}_p.npy") if os.path.isdir(os.path.join(os.path.dirname(__file__), "oof")) else "/dev/null", np.r_[Op, Tp])
if a.clf_only: sys.exit()
o = hurdle_median(Op, OQ, ALPHAS); t = hurdle_median(Tp, TQ, ALPHAS)
os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "oof"), exist_ok=True)
save_oof(a.name, o, t)
np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "oof", f"{a.name}_Q.npy"), np.r_[OQ, TQ])
m, s = fold_scores(y, o, F)
print(f"RESULT {a.name}: OOF {nmae(y, o):.5f} fold mean {m:.5f} sd {s:.5f}")

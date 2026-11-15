# Alternative P(y>0) classifiers (saved as oof/NAME_p.npy: OOF rows then fold-averaged test rows).
# usage: python t3/research/exp_clf.py NAME --algo cb|xgb [--depth 6] [--lr 0.05] [--seed 0] [--extra]
import sys, os, time, argparse, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load, extra_feats, NT
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss
ap = argparse.ArgumentParser(); ap.add_argument("name"); ap.add_argument("--algo", default="cb")
ap.add_argument("--depth", type=int, default=6); ap.add_argument("--lr", type=float, default=0.05); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--extra", action="store_true")
a = ap.parse_args()
OD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "oof")
D = load(); X, Xt = (extra_feats(D["X"], D["Xt"]) if a.extra else (D["X"], D["Xt"])); y, F = D["y"], D["folds"]; n = len(y)
Op = np.zeros(n); Tp = np.zeros(len(Xt)); t0 = time.time()
for k, (tr, va) in enumerate(F):
    t = (y[tr] > 0).astype(int); i, j = train_test_split(np.arange(len(tr)), test_size=0.1, random_state=a.seed, stratify=t)
    Xa = X.iloc[tr]
    if a.algo == "cb":
        from catboost import CatBoostClassifier
        m = CatBoostClassifier(iterations=20000, learning_rate=a.lr, depth=a.depth, l2_leaf_reg=3, od_type="Iter", od_wait=200,
                               random_seed=a.seed, thread_count=NT, verbose=False)
        m.fit(Xa.iloc[i], t[i], eval_set=(Xa.iloc[j], t[j]), use_best_model=True)
    else:
        import xgboost as xgb
        m = xgb.XGBClassifier(n_estimators=20000, learning_rate=a.lr, max_depth=a.depth, min_child_weight=5, subsample=0.8,
                              colsample_bytree=0.7, reg_lambda=2.0, tree_method="hist", n_jobs=NT, random_state=a.seed,
                              early_stopping_rounds=200, eval_metric="logloss")
        m.fit(Xa.iloc[i], t[i], eval_set=[(Xa.iloc[j], t[j])], verbose=False)
    Op[va] = m.predict_proba(X.iloc[va])[:, 1]; Tp += m.predict_proba(Xt)[:, 1] / 5
    print(f"fold {k} {time.time()-t0:.0f}s", flush=True)
np.save(os.path.join(OD, f"{a.name}_p.npy"), np.r_[Op, Tp])
print(f"RESULT {a.name}: AUC {roc_auc_score(y > 0, Op):.5f} logloss {log_loss(y > 0, Op):.5f}")

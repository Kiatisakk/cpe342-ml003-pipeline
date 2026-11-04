# In-sample ("train") score: fit classifier + CatBoost offset quantile model on ALL train rows, predict the SAME rows.
# Optimistic by construction (the model has seen these labels); shown only next to the honest OOF score to measure overfitting.
import sys, os, numpy as np, lightgbm as lgb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load, nmae, hurdle_median, extra_feats, prev_offset, NT
from catboost import CatBoostRegressor
ALPHAS = np.array([0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
D = load(); X, _ = extra_feats(D["X"], D["Xt"]); y = D["y"]
pos = y > 0
clf = lgb.LGBMClassifier(objective="binary", learning_rate=0.03, num_leaves=63, min_child_samples=40, subsample=0.8, subsample_freq=1,
                         colsample_bytree=0.7, reg_lambda=2.0, n_estimators=700, verbose=-1, n_jobs=NT, random_state=0).fit(X, pos.astype(int))
p = clf.predict_proba(X)[:, 1]
CFG = {"final (d4, l2 3)": dict(depth=4, l2_leaf_reg=3, iters=800),
       "optuna t20 (d3)": dict(depth=3, l2_leaf_reg=5.683, random_strength=0.0514, bagging_temperature=0.273, rsm=0.745, border_count=128, iters=1100)}
loss = "MultiQuantile:alpha=" + ",".join(str(x) for x in ALPHAS)
Xp = X[pos]; r = np.log(y[pos]) - prev_offset(Xp)
for nm, c in CFG.items():
    c = dict(c); it = c.pop("iters")    # ~ mean best iteration of the 5 CV folds
    m = CatBoostRegressor(loss_function=loss, iterations=it, learning_rate=0.05, random_seed=0, thread_count=NT, verbose=False, **c).fit(Xp, r)
    Q = np.exp(np.sort(m.predict(X), 1) + prev_offset(X)[:, None])
    print(f"RESULT in-sample {nm}: train NMAE score {nmae(y, hurdle_median(p, Q, ALPHAS)):.5f}")

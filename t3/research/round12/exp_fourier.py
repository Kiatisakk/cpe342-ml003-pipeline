# Paired test: add 2-harmonic Fourier features of seasonal_spending_pattern (month 0..11) vs control, same folds, seed and recipe.
import sys, os, time, numpy as np
sys.path.insert(0, os.path.abspath("t3/research"))
from common import load, nmae, hurdle_median, extra_feats, prev_offset
from catboost import CatBoostRegressor
import lightgbm as lgb
from sklearn.model_selection import train_test_split
AL = np.array([.02,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5]); LOSS = "MultiQuantile:alpha=" + ",".join(map(str, AL)); NT = 4
D = load(); X0, T0 = extra_feats(D["X"], D["Xt"]); y, F = D["y"], D["folds"]; n = len(y)
def fill(Z):
    Z = Z.copy(); m = Z.seasonal_spending_pattern
    for h in (1, 2):
        Z[f"sea_cos{h}"] = np.cos(2*np.pi*h*m/12); Z[f"sea_sin{h}"] = np.sin(2*np.pi*h*m/12)
    return Z
def _unused(Z):
    Z = Z.copy(); m = Z.is_premium_member.isna() & Z.vip_status.notna()
    Z.loc[m, "is_premium_member"] = (Z.vip_status[m] >= 3).astype(float); return Z
OUT = {}
for arm, (X, T) in {"control": (X0, T0), "fourier": (fill(X0), fill(T0))}.items():
    t0 = time.time(); p = np.zeros(n); Q = np.zeros((n, len(AL))); tp = np.zeros(len(T)); tQ = np.zeros((len(T), len(AL)))
    C = [c for c in X.columns if X[c].dtype == object]
    Xl, Tl = X.copy(), T.copy()
    for c in C:
        Xl[c] = Xl[c].astype("category"); Tl[c] = Tl[c].astype("category")
    for k, (tr, va) in enumerate(F):
        i, j = train_test_split(tr, test_size=.1, random_state=0)
        g = lgb.LGBMClassifier(n_estimators=5000, learning_rate=.03, num_leaves=63, subsample=.8, subsample_freq=1, colsample_bytree=.7, n_jobs=NT, random_state=0, verbose=-1)
        g.fit(Xl.iloc[i], y[i] > 0, eval_set=[(Xl.iloc[j], y[j] > 0)], callbacks=[lgb.early_stopping(100, verbose=False)])
        p[va] = g.predict_proba(Xl.iloc[va])[:, 1]; tp += g.predict_proba(Tl)[:, 1] / 5
        pos = y[tr] > 0; Xp = X.iloc[tr][pos]; r = np.log(y[tr][pos]) - prev_offset(Xp)
        a, b = train_test_split(np.arange(len(r)), test_size=.1, random_state=0)
        m = CatBoostRegressor(loss_function=LOSS, iterations=20000, learning_rate=.05, depth=4, l2_leaf_reg=3, od_type="Iter", od_wait=200, random_seed=0, thread_count=NT, verbose=False, cat_features=C)
        m.fit(Xp.iloc[a], r[a], eval_set=(Xp.iloc[b], r[b]), use_best_model=True)
        Q[va] = np.exp(np.sort(m.predict(X.iloc[va]), 1) + prev_offset(X.iloc[va])[:, None])
        tQ += np.exp(np.sort(m.predict(T), 1) + prev_offset(T)[:, None]) / 5
        print(arm, "fold", k, round(time.time() - t0), "s", flush=True)
    o = hurdle_median(p, Q, AL); t = hurdle_median(tp, tQ, AL); OUT[arm] = o
    np.save(f"t3/research/round12/{arm}_oof.npy", o); np.save(f"t3/research/round12/{arm}_test.npy", t)
    print("RESULT", arm, round(nmae(y, o), 5), flush=True)
d = [nmae(y[b], OUT["fourier"][b]) - nmae(y[b], OUT["control"][b]) for _, b in F]
print("PAIRED fourier-control per fold", np.round(d, 5), "mean", round(float(np.mean(d)), 5), "sd", round(float(np.std(d)), 5))
print("rows changed", int((OUT["fourier"] != OUT["control"]).sum()))

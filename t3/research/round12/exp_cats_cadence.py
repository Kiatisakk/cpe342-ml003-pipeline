# Paired test vs round10 control (same recipe, seed and folds; control reproduced exactly in round12: 0.79521).
# arm "cats": CatBoost native categorical (ordered target stats) for 5 discrete columns.
# arm "cadence": add purchase-cadence features tau = age/(tx+1), strain = (dslp - tau)/sqrt(tau+1), lapsed = dslp/(tau+1).
import sys, os, time, numpy as np
sys.path.insert(0, os.path.abspath("t3/research"))
from common import load, nmae, hurdle_median, extra_feats, prev_offset
from catboost import CatBoostRegressor
import lightgbm as lgb
from sklearn.model_selection import train_test_split
AL = np.array([.02,.05,.1,.15,.2,.25,.3,.35,.4,.45,.5]); LOSS = "MultiQuantile:alpha=" + ",".join(map(str, AL)); NT = 4
D = load(); X0, T0 = extra_feats(D["X"], D["Xt"]); y, F = D["y"], D["folds"]; n = len(y)
CAT = ["seasonal_spending_pattern", "segment", "vip_status", "platform", "primary_game"]
def cad(Z):
    Z = Z.copy(); tau = Z.account_age_days / (Z.total_transactions + 1)
    Z["cad_tau"] = tau; Z["cad_strain"] = (Z.days_since_last_purchase - tau) / np.sqrt(tau + 1); Z["cad_lapsed"] = Z.days_since_last_purchase / (tau + 1); return Z
def catstr(Z):
    Z = Z.copy()
    for c in CAT: Z[c + "_c"] = Z[c].fillna(-1).astype(int).astype(str)
    return Z
ctrl = np.load("t3/research/round10/control_oof.npy"); print("control", round(nmae(y, ctrl), 5), flush=True)
for arm, (X, T, C) in {"cats": (catstr(X0), catstr(T0), [c + "_c" for c in CAT]), "cadence": (cad(X0), cad(T0), [])}.items():
    t0 = time.time(); p = np.zeros(n); Q = np.zeros((n, len(AL))); tp = np.zeros(len(T)); tQ = np.zeros((len(T), len(AL)))
    Xl, Tl = X.drop(columns=C), T.drop(columns=C)   # classifier unchanged (numeric) so the arm isolates the amount model / features
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
    o = hurdle_median(p, Q, AL); t = hurdle_median(tp, tQ, AL)
    np.save(f"t3/research/round12/{arm}_oof.npy", o); np.save(f"t3/research/round12/{arm}_test.npy", t)
    d = [nmae(y[b], o[b]) - nmae(y[b], ctrl[b]) for _, b in F]
    print("RESULT", arm, round(nmae(y, o), 5), "paired per fold", np.round(d, 5), "mean", round(float(np.mean(d)), 5), "sd", round(float(np.std(d)), 5), flush=True)

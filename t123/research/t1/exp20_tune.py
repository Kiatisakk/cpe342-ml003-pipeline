"""Hyper-parameter tuning round 2 (Optuna; allowed by course rules) for LGB / XGB / CatBoost on fe_v2.

Selection protocol (to avoid overfitting the evaluation folds):
  * Optuna objective = OOF AUC on a SEPARATE 3-fold split (seed 99), fixed lr (0.1 LGB/XGB, 0.15 Cat) for speed.
  * Top-3 distinct trials per model are then re-evaluated with the usual 5-fold x seeds 42/7/2024 at lr 0.05,
    against the baseline-tuned params; nested-threshold F2m and AUC reported. OOFs saved for blending.
usage: python exp20_tune.py lgb [n_trials]   |   xgb   |   cat
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import *
import optuna, lightgbm as lgb, xgboost as xgb, catboost as cb
optuna.logging.set_verbosity(optuna.logging.WARNING)
lab, unl, te, ids = load(); y = lab[TARGET].astype(int).values
X = fe_v2(lab, pd.concat([lab, unl]))
TF = list(StratifiedKFold(3, shuffle=True, random_state=99).split(X, y))
MODEL = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else {"lgb": 120, "xgb": 50, "cat": 40}[MODEL]


def space(t):
    if MODEL == "lgb":
        return dict(num_leaves=t.suggest_int("num_leaves", 7, 400, log=True), max_depth=t.suggest_categorical("max_depth", [-1, 6, 8, 10, 12]),
                    min_child_samples=t.suggest_int("min_child_samples", 3, 500, log=True),
                    min_sum_hessian_in_leaf=t.suggest_float("min_sum_hessian_in_leaf", 1e-4, 10, log=True),
                    subsample=t.suggest_float("subsample", 0.4, 1.0), subsample_freq=1,
                    colsample_bytree=t.suggest_float("colsample_bytree", 0.15, 1.0), feature_fraction_bynode=t.suggest_float("feature_fraction_bynode", 0.3, 1.0),
                    reg_lambda=t.suggest_float("reg_lambda", 1e-3, 200, log=True), reg_alpha=t.suggest_float("reg_alpha", 1e-3, 50, log=True),
                    min_split_gain=t.suggest_float("min_split_gain", 1e-4, 2, log=True), max_bin=t.suggest_categorical("max_bin", [63, 127, 255, 511, 1023]),
                    path_smooth=t.suggest_float("path_smooth", 1e-3, 100, log=True), verbose=-1, n_jobs=4)
    if MODEL == "xgb":
        return dict(max_depth=t.suggest_int("max_depth", 3, 12), min_child_weight=t.suggest_float("min_child_weight", 0.1, 200, log=True),
                    subsample=t.suggest_float("subsample", 0.4, 1.0), colsample_bytree=t.suggest_float("colsample_bytree", 0.2, 1.0),
                    colsample_bynode=t.suggest_float("colsample_bynode", 0.3, 1.0),
                    reg_lambda=t.suggest_float("reg_lambda", 1e-3, 200, log=True), reg_alpha=t.suggest_float("reg_alpha", 1e-3, 50, log=True),
                    gamma=t.suggest_float("gamma", 1e-4, 10, log=True), max_bin=t.suggest_categorical("max_bin", [64, 128, 256, 512]),
                    grow_policy=t.suggest_categorical("grow_policy", ["depthwise", "lossguide"]), tree_method="hist", n_jobs=4)
    p = dict(depth=t.suggest_int("depth", 3, 10), l2_leaf_reg=t.suggest_float("l2_leaf_reg", 0.1, 100, log=True),
             random_strength=t.suggest_float("random_strength", 1e-3, 20, log=True), border_count=t.suggest_categorical("border_count", [64, 128, 254]),
             bootstrap_type=t.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli"]),
             grow_policy=t.suggest_categorical("grow_policy", ["SymmetricTree", "Depthwise", "Lossguide"]), thread_count=4)
    if p["bootstrap_type"] == "Bayesian": p["bagging_temperature"] = t.suggest_float("bagging_temperature", 0, 5)
    else: p["subsample"] = t.suggest_float("subsample", 0.5, 1.0)
    if p["grow_policy"] != "SymmetricTree": p["min_data_in_leaf"] = t.suggest_int("min_data_in_leaf", 1, 300, log=True)
    if p["grow_policy"] == "Lossguide": p["max_leaves"] = t.suggest_int("max_leaves", 8, 128, log=True)
    return p


def fit_pred(p, lr, Xa, ya, Xv, yv, Xb, rs, rounds=8000, es=150):
    if MODEL == "lgb":
        m = lgb.LGBMClassifier(n_estimators=rounds, learning_rate=lr, random_state=rs, **p)
        m.fit(Xa, ya, eval_set=[(Xv, yv)], eval_metric="auc", callbacks=[lgb.early_stopping(es, verbose=False)])
    elif MODEL == "xgb":
        m = xgb.XGBClassifier(n_estimators=rounds, learning_rate=lr, early_stopping_rounds=es, eval_metric="auc", random_state=rs, **p)
        m.fit(Xa, ya, eval_set=[(Xv, yv)], verbose=False)
    else:
        m = cb.CatBoostClassifier(iterations=rounds, learning_rate=lr, eval_metric="AUC", od_type="Iter", od_wait=es, random_seed=rs, verbose=False, **p)
        m.fit(Xa, ya, eval_set=(Xv, yv), use_best_model=True)
    return m.predict_proba(Xb)[:, 1]


def objective(t):
    p = space(t); oof = np.zeros(len(y)); lr = 0.15 if MODEL == "cat" else 0.1
    for k, (a, b) in enumerate(TF):
        r = np.random.default_rng(k); a = r.permutation(a); c = int(0.9 * len(a))
        oof[b] = fit_pred(p, lr, X.iloc[a[:c]], y[a[:c]], X.iloc[a[c:]], y[a[c:]], X.iloc[b], k, es=100)
    return roc_auc_score(y, oof)


BASE = {"lgb": {k: v for k, v in LGB_BASE.items() if k not in ("verbose", "n_jobs")},
        "xgb": {k: v for k, v in XGB_BASE.items() if k not in ("tree_method", "n_jobs")},
        "cat": {k: v for k, v in CAT_BASE.items() if k != "thread_count"}}
if __name__ == "__main__":
    T = Timer()
    st = optuna.create_study(direction="maximize", study_name=f"{MODEL}_r2", storage=f"sqlite:///{OUT}/optuna_{MODEL}_r2.db", load_if_exists=True,
                             sampler=optuna.samplers.TPESampler(seed=1, multivariate=True, n_startup_trials=20))
    if len(st.trials) == 0:
        bp = dict(BASE[MODEL]); bp.pop("subsample_freq", None)
        if MODEL == "lgb": bp.update(max_depth=-1, min_sum_hessian_in_leaf=1e-3, feature_fraction_bynode=1.0, path_smooth=1e-3)
        if MODEL == "xgb": bp.update(colsample_bynode=1.0)
        if MODEL == "cat": bp.update(grow_policy="SymmetricTree")
        st.enqueue_trial(bp)
    st.optimize(objective, n_trials=max(0, N - len(st.trials)), gc_after_trial=True,
                callbacks=[lambda s, t: print(f"[{MODEL}] trial {t.number} AUC {t.value:.5f} | best {s.best_value:.5f} | {T()}", flush=True) if t.number % 10 == 0 else None])
    done = sorted([t for t in st.trials if t.value is not None], key=lambda t: -t.value)
    print(f"[{MODEL}] tuning done {T()}: baseline-params trial {st.trials[0].value:.5f}, best {done[0].value:.5f}", flush=True)
    cands = [("baseline", None)] + [(f"top{i+1}(t{t.number})", t) for i, t in enumerate(done[:3])]
    for nm, t in cands:
        if t is None: p = dict(BASE[MODEL]); p.update({"lgb": dict(verbose=-1, n_jobs=4), "xgb": dict(tree_method="hist", n_jobs=4), "cat": dict(thread_count=4)}[MODEL])
        else: p = space(optuna.trial.FixedTrial(t.params))
        def run(s, p=p):
            oof = np.zeros(len(y))
            for k, (a, b) in enumerate(folds_for(y, s)):
                r = np.random.default_rng(s + k); a = r.permutation(a); c = int(0.9 * len(a))
                oof[b] = fit_pred(p, 0.05, X.iloc[a[:c]], y[a[:c]], X.iloc[a[c:]], y[a[c:]], X.iloc[b], s + k, es=200)
            return oof
        agg, o = multi_seed(f"exp20 {MODEL} {nm} lr0.05", y, run, extra=dict(tune_auc=None if t is None else round(t.value, 5)))
        np.save(f"{OUT}/exp20_{MODEL}_{nm.split('(')[0]}.npy", np.array(o))
        if t is not None: json.dump(t.params, open(f"{OUT}/exp20_{MODEL}_{nm.split('(')[0]}_params.json", "w"), indent=1)
    print("done", T())

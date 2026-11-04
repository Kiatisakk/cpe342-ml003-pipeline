"""Optuna re-tune of LightGBM on fe_v2. Objective = AUC on a *separate* 3-fold split (seed 99, lr 0.08), so the
5-fold evaluation seeds (42/7/2024) are not used for selection. Best params then evaluated with multi_seed at lr 0.02."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import *
import optuna, lightgbm as lgb
optuna.logging.set_verbosity(optuna.logging.WARNING)
lab, unl, te, ids = load(); y = lab[TARGET].astype(int).values
X = fe_v2(lab, pd.concat([lab, unl]))
tune_folds = list(StratifiedKFold(3, shuffle=True, random_state=99).split(X, y))
N = int(os.environ.get("N_TRIALS", 60))


def objective(t):
    p = dict(num_leaves=t.suggest_int("num_leaves", 7, 255, log=True), min_child_samples=t.suggest_int("min_child_samples", 5, 400, log=True),
             subsample=t.suggest_float("subsample", 0.5, 1.0), subsample_freq=1, colsample_bytree=t.suggest_float("colsample_bytree", 0.2, 1.0),
             reg_lambda=t.suggest_float("reg_lambda", 1e-3, 100, log=True), reg_alpha=t.suggest_float("reg_alpha", 1e-3, 30, log=True),
             min_split_gain=t.suggest_float("min_split_gain", 1e-4, 1, log=True), max_bin=t.suggest_categorical("max_bin", [63, 127, 255, 511]),
             path_smooth=t.suggest_float("path_smooth", 1e-3, 50, log=True), verbose=-1, n_jobs=4)
    oof = np.zeros(len(y))
    for k, (a, b) in enumerate(tune_folds):
        rng = np.random.default_rng(k); a = rng.permutation(a); cut = int(0.9 * len(a))
        m = lgb.LGBMClassifier(n_estimators=4000, learning_rate=0.08, random_state=k, **p)
        m.fit(X.iloc[a[:cut]], y[a[:cut]], eval_set=[(X.iloc[a[cut:]], y[a[cut:]])], eval_metric="auc", callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[b] = m.predict_proba(X.iloc[b])[:, 1]
    return roc_auc_score(y, oof)


st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0, multivariate=True, n_startup_trials=15),
                         storage=f"sqlite:///{OUT}/optuna_lgb_v2.db", study_name="lgb_v2", load_if_exists=True)
# baseline params as a starting point
st.enqueue_trial(dict(num_leaves=39, min_child_samples=8, subsample=0.814, colsample_bytree=0.455, reg_lambda=4.01, reg_alpha=0.0045,
                      min_split_gain=0.02, max_bin=511, path_smooth=1e-3))
st.optimize(objective, n_trials=max(0, N - len(st.trials)))
print("best 3-fold AUC", st.best_value, "baseline-params trial", st.trials[0].value, st.best_params, flush=True)
bp = dict(st.best_params); bp["subsample_freq"] = 1
json.dump(bp, open(f"{OUT}/lgb_v2_params.json", "w"), indent=1)
a, o = multi_seed("exp10 lgb fe_v2 Optuna-tuned lr0.02", y, lambda s: cv_lgb(X, y, seed=s, lr=0.02, params=bp)[0]); np.save(f"{OUT}/exp10_lgb_tuned.npy", np.array(o))

# Optuna (TPE) hyper-parameter search for the CatBoost offset multi-quantile amount model.
# HONEST protocol: each trial is scored ONLY on folds 0 and 1 of the v4 split (hurdle NMAE with the fixed 3-seed classifier p);
# folds 2-4 are never seen during tuning and give the unbiased estimate when the best params are refit on all folds.
# usage: python t123/research/t3/tune_cb.py [--trials 40] [--timeout 7200]
import sys, os, time, argparse, json, numpy as np, optuna
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load, nmae, hurdle_median, extra_feats, prev_offset, NT, CACHE, HERE
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
ap = argparse.ArgumentParser(); ap.add_argument("--trials", type=int, default=40); ap.add_argument("--timeout", type=int, default=7200)
a = ap.parse_args()
ALPHAS = np.array([0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
OD = os.path.join(HERE, "oof")
D = load(); X, Xt = extra_feats(D["X"], D["Xt"]); y, F = D["y"], D["folds"]; n = len(y)
P = np.mean([np.load(os.path.join(OD, f"{q}_p.npy")) for q in ("lgbh_log", "clf_s1", "clf_s2")], 0)[:n]
TUNE_FOLDS = [0, 1]
loss = "MultiQuantile:alpha=" + ",".join(str(x) for x in ALPHAS)


def run_fold(k, prm):
    tr, va = F[k]; ya = y[tr]; pos = ya > 0; Xp = X.iloc[tr][pos]; r = np.log(ya[pos]) - prev_offset(Xp)
    i, j = train_test_split(np.arange(len(r)), test_size=0.1, random_state=0)
    extra = {"min_data_in_leaf": prm["mdl"]} if prm["grow"] != "SymmetricTree" else {}
    m = CatBoostRegressor(loss_function=loss, iterations=20000, learning_rate=0.05, depth=prm["depth"], l2_leaf_reg=prm["l2"],
                          random_strength=prm["rs"], bagging_temperature=prm["bt"], rsm=prm["rsm"], border_count=prm["border"],
                          grow_policy=prm["grow"], od_type="Iter", od_wait=200, random_seed=0, thread_count=NT, verbose=False, **extra)
    m.fit(Xp.iloc[i], r[i], eval_set=(Xp.iloc[j], r[j]), use_best_model=True)
    Q = np.exp(np.sort(m.predict(X.iloc[va]), 1) + prev_offset(X.iloc[va])[:, None])
    o = hurdle_median(P[va], Q, ALPHAS)
    return np.abs(y[va] - o).sum(), y[va].sum(), m.get_best_iteration()


def objective(trial):
    prm = dict(depth=trial.suggest_int("depth", 3, 8), l2=trial.suggest_float("l2", 0.5, 50, log=True),
               rs=trial.suggest_float("rs", 0.05, 10, log=True), bt=trial.suggest_float("bt", 0.0, 2.0),
               rsm=trial.suggest_float("rsm", 0.4, 1.0), border=trial.suggest_categorical("border", [64, 128, 254]),
               grow=trial.suggest_categorical("grow", ["SymmetricTree", "Depthwise"]), mdl=trial.suggest_int("mdl", 1, 300, log=True))
    E = Y = 0.0; its = []
    for step, k in enumerate(TUNE_FOLDS):
        e, s, it = run_fold(k, prm); E += e; Y += s; its.append(it)
        trial.report(1 / (1 + E / Y), step)
        if trial.should_prune(): raise optuna.TrialPruned()
    trial.set_user_attr("iters", its)
    return 1 / (1 + E / Y)


study = optuna.create_study(direction="maximize", study_name="cb_offset_q", storage=f"sqlite:///{CACHE}/optuna_cb.db", load_if_exists=True,
                            sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=8),
                            pruner=optuna.pruners.MedianPruner(n_startup_trials=6, n_warmup_steps=0))
if len(study.trials) == 0:   # enqueue the current final params as the reference trial
    study.enqueue_trial(dict(depth=4, l2=3.0, rs=1.0, bt=1.0, rsm=1.0, border=254, grow="SymmetricTree", mdl=1))
t0 = time.time()
study.optimize(objective, n_trials=a.trials, timeout=a.timeout,
               callbacks=[lambda s, t: print(f"trial {t.number} {t.state.name} value {t.value} params {t.params} | best {s.best_value:.5f} ({time.time()-t0:.0f}s)", flush=True)])
print("BEST", study.best_value, study.best_params)
json.dump({"best_value_folds01": study.best_value, "best_params": study.best_params, "n_trials": len(study.trials)},
          open(os.path.join(HERE, "tune_cb_best.json"), "w"), indent=1)

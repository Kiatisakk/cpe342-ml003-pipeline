"""E17b: NESTED Optuna tuning of LightGBM multiclass (features only).
Per outer fold: outer-train A is split 80/20 (stratified); Optuna (TPE) minimises log-loss on the 20% part
(early stopping on a 10% slice of the 80% part, lr 0.1 while tuning). Best params are refit on A with lr 0.03
(early stopping on 10% of A) and scored on the outer fold. The outer fold is never seen by the tuner.
usage: python t123/research/t2/e17_tune_lgbm.py <seed> [n_trials]"""
import sys, json; sys.path.insert(0, 't123/research/t2')
from runner import *
import lightgbm as lgb, optuna
from sklearn.model_selection import train_test_split
from e05_lgbm import raw
optuna.logging.set_verbosity(optuna.logging.WARNING)
SEED = int(sys.argv[1]); NTR = int(sys.argv[2]) if len(sys.argv) > 2 else 25
TH = int(os.environ.get('OMP_NUM_THREADS', 1))
BASE = dict(objective='multiclass', num_class=K, verbose=-1, num_threads=TH, seed=SEED, bagging_freq=1)


def train(p, Xa, ya, Xv, yv, lr):
    return lgb.train({**BASE, **p, 'learning_rate': lr}, lgb.Dataset(Xa, ya), 5000, valid_sets=[lgb.Dataset(Xv, yv)],
                     callbacks=[lgb.early_stopping(100, verbose=False)])


def run_seed(seed):
    oof = np.zeros((N, K)); te_ = np.zeros((NT, K)); chosen = []
    for fo, (A, Bo) in enumerate(folds(seed)):
        Xt, Xe = raw(A)
        A1, A2 = train_test_split(A, test_size=0.2, stratify=y[A], random_state=fo)
        A1t, A1v = train_test_split(A1, test_size=0.1, stratify=y[A1], random_state=fo)
        def obj(tr_):
            p = dict(num_leaves=tr_.suggest_int('num_leaves', 4, 64, log=True), min_child_samples=tr_.suggest_int('min_child_samples', 20, 800, log=True),
                     feature_fraction=tr_.suggest_float('feature_fraction', 0.3, 1.0), bagging_fraction=tr_.suggest_float('bagging_fraction', 0.5, 1.0),
                     lambda_l2=tr_.suggest_float('lambda_l2', 1e-3, 100, log=True), lambda_l1=tr_.suggest_float('lambda_l1', 1e-3, 10, log=True),
                     max_bin=tr_.suggest_categorical('max_bin', [63, 255, 511]), min_split_gain=tr_.suggest_float('min_split_gain', 0, 0.5))
            m = train(p, Xt[A1t], y[A1t], Xt[A1v], y[A1v], 0.1)
            return log_loss(y[A2], m.predict(Xt[A2]))
        st = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=fo))
        st.enqueue_trial(dict(num_leaves=15, min_child_samples=100, feature_fraction=0.7, bagging_fraction=0.8, lambda_l2=5.0, lambda_l1=1e-3, max_bin=255, min_split_gain=0.0))
        st.optimize(obj, n_trials=NTR); p = st.best_params; chosen.append(p)
        At, Av = train_test_split(A, test_size=0.1, stratify=y[A], random_state=0)
        m = train(p, Xt[At], y[At], Xt[Av], y[Av], 0.03)
        oof[Bo] = m.predict(Xt[Bo]); te_ += m.predict(Xe) / 5
        print(f'seed {seed} fold {fo}: tuner LL default {st.trials[0].value:.4f} best {st.best_value:.4f} | outer F1 {f1m(y[Bo], oof[Bo]):.4f}', flush=True)
    np.savez_compressed(f'{CACHE}/P_lgbm_tuned_s{seed}.npz', oof=oof.astype(np.float32), te=te_)
    json.dump(chosen, open(f'{CACHE}/lgbm_tuned_params_s{seed}.json', 'w'), indent=1)
    print(f'seed {seed}: tuned LGBM F1 {f1m(y, oof):.4f} LL {log_loss(y, oof):.4f} (untuned 0.7509 / 0.6173 on seed 42)', flush=True)


if __name__ == '__main__':
    run_seed(SEED)

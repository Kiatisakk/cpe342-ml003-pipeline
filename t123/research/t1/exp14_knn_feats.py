"""Out-of-fold kNN target features (nested): space = rank-gauss of 14 signal features (age/level posterior means,
lps dropped), NaN -> 0 (mean). Per outer fold: training rows get kNN rates from an inner 5-fold, held-out rows from
all outer-train rows. Then LGB on fe_v2 + kNN features. Quantile transformer fitted on TRAIN rows only."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from sklearn.preprocessing import QuantileTransformer
from sklearn.neighbors import NearestNeighbors
import lightgbm as lgb
lab, unl, te, ids = load(); y = lab[TARGET].astype(int).values
ref = pd.concat([lab, unl]); X = fe_v2(lab, ref); z = fe_zone_post(lab, ref)
cols = [c for c in SIGNAL if c not in ("account_age_days", "level", "level_progression_speed")]
S = pd.concat([lab[cols], z[["age_post_mean", "lvl_post_mean"]]], axis=1).values
qt = QuantileTransformer(output_distribution="normal", random_state=0).fit(np.vstack([S, pd.concat([unl[cols], fe_zone_post(unl, ref)[["age_post_mean", "lvl_post_mean"]]], axis=1).values]))
E = np.nan_to_num(qt.transform(S), nan=0.0).astype(np.float32)
KS = (25, 100, 400)


def knn_rates(Etr, ytr, Eq, exclude_self=False):
    nn = NearestNeighbors(n_neighbors=max(KS) + int(exclude_self), n_jobs=4).fit(Etr)
    idx = nn.kneighbors(Eq, return_distance=False)
    if exclude_self: idx = idx[:, 1:]
    Y = ytr[idx]
    return np.column_stack([Y[:, :k].mean(1) for k in KS])


def run(seed):
    oof = np.zeros(len(y))
    for k, (a, b) in enumerate(folds_for(y, seed)):
        Ka = np.zeros((len(a), len(KS)))
        for ia, ib in StratifiedKFold(5, shuffle=True, random_state=seed + 100 + k).split(a, y[a]):
            Ka[ib] = knn_rates(E[a[ia]], y[a[ia]], E[a[ib]])
        Kb = knn_rates(E[a], y[a], E[b])
        F = lambda rows, K: pd.concat([X.iloc[rows].reset_index(drop=True), pd.DataFrame(K, columns=[f"knn{q}" for q in KS])], axis=1)
        Xa, Xb = F(a, Ka), F(b, Kb)
        rng = np.random.default_rng(seed + k); p = rng.permutation(len(a)); cut = int(0.9 * len(a))
        m = lgb.LGBMClassifier(n_estimators=5000, learning_rate=0.05, random_state=seed + k, **LGB_BASE)
        m.fit(Xa.iloc[p[:cut]], y[a][p[:cut]], eval_set=[(Xa.iloc[p[cut:]], y[a][p[cut:]])], eval_metric="auc", callbacks=[lgb.early_stopping(200, verbose=False)])
        oof[b] = m.predict_proba(Xb)[:, 1]
        print(f"  fold {k} knn100 auc {roc_auc_score(y[b], Kb[:, 1]):.4f}", flush=True)
    return oof


seeds = tuple(int(s) for s in sys.argv[1:]) or SEEDS
a, o = multi_seed("exp14 lgb fe_v2 + nested OOF kNN rates (k 25/100/400)", y, run, seeds=seeds, save=len(seeds) == 3)
if len(seeds) == 3: np.save(f"{OUT}/exp14.npy", np.array(o))

"""MCAR masking augmentation. Missingness is MCAR (cheater rate | NaN == base rate for every column), so copies
of TRAINING-fold rows with extra random NaNs follow the same joint law as genuinely-incomplete rows.
Per outer fold: n_copies masked copies of the fold-train rows (each signal value hidden with prob = rate * its
observed missing rate), fe_v2 recomputed on the masked raw data; early-stopping holdout and held-out fold untouched."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import *
import lightgbm as lgb
lab, unl, te, ids = load(); y = lab[TARGET].astype(int).values
ref = pd.concat([lab, unl]); X = fe_v2(lab, ref)
NC = int(os.environ.get("NC", 2)); RATE = float(os.environ.get("RATE", 1.0)); LR = float(os.environ.get("LR", 0.05))
WAUG = float(os.environ.get("WAUG", 1.0))
miss_rate = lab[SIGNAL].isna().mean().values


def masked_copy(df, rng):
    d = df.copy(); M = rng.random((len(d), len(SIGNAL))) < miss_rate[None, :] * RATE
    vals = d[SIGNAL].values.copy(); vals[M] = np.nan; d[SIGNAL] = vals
    return fe_v2(d, ref)


def run(seed, Xt=None):
    oof = np.zeros(len(y)); tp = None if Xt is None else np.zeros(len(Xt))
    for k, (a, b) in enumerate(folds_for(y, seed)):
        rng = np.random.default_rng(seed + k); a = rng.permutation(a); cut = int(0.9 * len(a)); tr, va = a[:cut], a[cut:]
        parts = [X.iloc[tr]] + [masked_copy(lab.iloc[tr], rng) for _ in range(NC)]
        Xa = pd.concat(parts, ignore_index=True); ya = np.tile(y[tr], NC + 1)
        w = np.concatenate([np.ones(len(tr))] + [np.full(len(tr), WAUG)] * NC)
        m = lgb.LGBMClassifier(n_estimators=8000, learning_rate=LR, random_state=seed + k, **LGB_BASE)
        m.fit(Xa, ya, sample_weight=w, eval_set=[(X.iloc[va], y[va])], eval_metric="auc", callbacks=[lgb.early_stopping(200, verbose=False)])
        oof[b] = m.predict_proba(X.iloc[b])[:, 1]
        if Xt is not None: tp += m.predict_proba(Xt)[:, 1] / 5
    return oof if Xt is None else (oof, tp)


if __name__ == "__main__":
    tag = f"NC{NC}_rate{RATE}_w{WAUG}_lr{LR}"
    a, o = multi_seed(f"exp16 lgb fe_v2 + MCAR mask augmentation ({tag})", y, run)
    np.save(f"{OUT}/exp16_{tag}.npy", np.array(o))
    oo = np.array(o)
    for f in ["reports_received", "friend_network_size", "account_age_days", "crosshair_placement"]:
        m = lab[f].isna().values
        print(f, "AUC on rows missing it (seed42):", round(roc_auc_score(y[m], oo[0][m]), 4))

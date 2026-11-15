"""Mixture naive Bayes (from scratch, EM) on 14 features; NaN marginalised. Screens K / bins / smoothing."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from mixnb import Binner, MixNB


def nb_matrix(df, ref):
    z = fe_zone_post(df, ref)
    cols = [c for c in SIGNAL if c not in ("level_progression_speed", "account_age_days", "level")]
    M = df[cols].copy(); M["age"] = z["age_post_mean"]; M["lvl"] = z["lvl_post_mean"]
    return M


def cv_mixnb(M, y, seed, K=(4, 4), n_bins=32, age_bins=120, alpha=1.0, smooth=0, n_iter=60, Mu=None, nb_seeds=1, Mt=None):
    X = M.values.astype(float)
    oof = np.zeros(len(y)); tp = np.zeros(len(Mt)) if Mt is not None else None
    for a, b in folds_for(y, seed):
        ref = X[a] if Mu is None else np.vstack([X[a], Mu.values])
        bn = Binner(n_bins).fit(ref)
        j = list(M.columns).index("age"); bn.edges[j] = Binner(age_bins).fit(ref[:, [j]]).edges[0]; bn.sizes = [len(e) - 1 for e in bn.edges]
        C = bn.transform(X); Cu = bn.transform(Mu.values) if Mu is not None else None
        for s in range(nb_seeds):
            m = MixNB(K, alpha, n_iter, seed=seed * 10 + s, smooth=smooth).fit(C[a], y[a], Cu)
            oof[b] += m.decision(C[b]) / nb_seeds
            if Mt is not None: tp += m.decision(bn.transform(Mt.values)) / nb_seeds / 5
    return oof, tp


if __name__ == "__main__":
    lab, unl, te, ids = load(); y = lab[TARGET].astype(int).values
    ref = pd.concat([lab, unl]); M = nb_matrix(lab, ref)
    grid = [dict(K=(1, 1)), dict(K=(3, 2)), dict(K=(6, 3)), dict(K=(8, 4)), dict(K=(12, 6))]
    for g in grid:
        T = Timer(); o, _ = cv_mixnb(M, y, 42, **g)
        report(f"exp04 mixnb {g}", y, o, extra=dict(time=T()))

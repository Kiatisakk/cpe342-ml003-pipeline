# Shared data prep for T3 research. Run from repo root.
# Reproduces t123/task3_hurdle_ziln_v4.py preprocessing exactly (exact fill, features, segment imputer
# fitted on TRAIN rows with known segment only) and its 5-fold scheme (StratifiedKFold seed 42 on
# zero / payer-quartile strata). test.csv is only transformed, never used for fitting anything.
import os, pickle, numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join("cpe342-kaggle-main", "03_regression")
CACHE = os.path.join(HERE, "cache"); os.makedirs(CACHE, exist_ok=True)
TARGET = "spending_30d"
NT = 4


def nmae(yt, yp):
    return 1 / (1 + np.abs(yt - yp).sum() / yt.sum())


def exact_fill(d):
    d = d.copy()
    for _ in range(3):
        h, tx, av = d["historical_spending"], d["total_transactions"], d["avg_transaction_value"]
        po, dr = d["purchases_on_discount"], d["discount_rate_used"]
        rules = [
            ("historical_spending", h.isna() & tx.notna() & av.notna(), tx * av),
            ("total_transactions", tx.isna() & h.notna() & av.notna() & (av > 0), h / av),
            ("avg_transaction_value", av.isna() & h.notna() & tx.notna() & (tx > 0), h / tx),
            ("purchases_on_discount", po.isna() & tx.notna() & dr.notna(), tx * dr),
            ("total_transactions", tx.isna() & po.notna() & dr.notna() & (dr > 0), po / dr),
            ("discount_rate_used", dr.isna() & po.notna() & tx.notna() & (tx > 0), po / tx),
        ]
        for col, m, val in rules:
            m = m & d[col].isna(); d.loc[m, col] = val[m]
    return d


def add_feats(d):
    d = d.copy(); e = 1.0
    d["hist_per_day"] = d["historical_spending"] / (d["account_age_days"] + e)
    d["tx_per_day"] = d["total_transactions"] / (d["account_age_days"] + e)
    d["exp_monthly_spend"] = d["avg_transaction_value"] * d["purchase_frequency"]
    d["prev_vs_hist"] = d["prev_month_spending"] / (d["historical_spending"] + e)
    d["prev_x_freq"] = d["prev_month_spending"] * d["purchase_frequency"]
    d["recency_ratio"] = d["days_since_last_purchase"] / (d["days_since_last_login"] + e)
    d["play_per_session"] = d["total_playtime_hours"] / (d["sessions_per_week"] + e)
    for c in ["historical_spending", "prev_month_spending", "avg_transaction_value"]:
        d[f"log_{c}"] = np.log1p(d[c].clip(lower=0))
    d["n_missing"] = d.isna().sum(axis=1)
    return d


def folds_v4(y, seed=42):
    from sklearn.model_selection import StratifiedKFold
    pos = y > 0
    strat = np.where(pos, 1 + pd.qcut(pd.Series(y).where(pos).rank(method="first"), 4, labels=False).fillna(-1).values, 0).astype(int)
    return list(StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(y)), strat))


def load():
    f = os.path.join(CACHE, "data_v4.pkl")
    if os.path.exists(f):
        return pickle.load(open(f, "rb"))
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    tr = pd.read_csv(os.path.join(DATA, "train.csv")); te = pd.read_csv(os.path.join(DATA, "test.csv"))
    y = tr[TARGET].values.astype(float)
    seg_raw_tr, seg_raw_te = tr["segment"].values.copy(), te["segment"].values.copy()
    tr, te = exact_fill(tr), exact_fill(te)
    X, Xt = add_feats(tr.drop(columns=[TARGET])), add_feats(te)
    FE = [c for c in Xt.columns if c != "player_id"]
    X, Xt = X[FE].copy(), Xt[FE].copy()
    SF = [c for c in FE if c != "segment"]
    known = X["segment"].notna().values
    sc = xgb.XGBClassifier(objective="multi:softprob", n_estimators=4000, learning_rate=0.02, early_stopping_rounds=100,
                           max_depth=6, subsample=0.8, colsample_bytree=0.7, tree_method="hist", random_state=42, n_jobs=NT)
    i_a, i_b = train_test_split(np.where(known)[0], test_size=0.1, random_state=42, stratify=X["segment"].values[known])
    sc.fit(X.iloc[i_a][SF], X["segment"].values[i_a].astype(int), eval_set=[(X.iloc[i_b][SF], X["segment"].values[i_b].astype(int))], verbose=False)
    print("segment imputer holdout acc", (sc.predict(X.iloc[i_b][SF]) == X["segment"].values[i_b]).mean())
    SP = {}
    for nm, D in (("tr", X), ("te", Xt)):
        miss = D["segment"].isna().values
        P = np.zeros((len(D), 4)); P[~miss, D.loc[~miss, "segment"].astype(int).values] = 1
        P[miss] = sc.predict_proba(D.loc[miss, SF])
        D["seg_imputed"] = miss.astype(int)
        D.loc[miss, "segment"] = P[miss].argmax(1)
        D["p_whale"] = P[:, 3]
        SP[nm] = P
    ids = np.array([f"ANS{i+1:05d}" for i in range(len(Xt))])
    out = dict(X=X, Xt=Xt, y=y, ids=ids, folds=folds_v4(y), segP=SP["tr"], segPt=SP["te"],
               seg_raw=seg_raw_tr, seg_raw_te=seg_raw_te)
    pickle.dump(out, open(f, "wb"))
    return out


def save_oof(name, oof, test):
    d = os.path.join(HERE, "oof"); os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, f"{name}_oof.npy"), oof); np.save(os.path.join(d, f"{name}_test.npy"), test)


def load_oof(name):
    d = os.path.join(HERE, "oof")
    return np.load(os.path.join(d, f"{name}_oof.npy")), np.load(os.path.join(d, f"{name}_test.npy"))


def fold_scores(y, p, folds):
    s = np.array([nmae(y[b], p[b]) for _, b in folds])
    return s.mean(), s.std()


def hurdle_median(p, Q, alphas, shift=0.0, kap=1.0, cap=300000):
    """MAE-optimal prediction from P(y>0)=p and positive-part quantiles Q (n x len(alphas)) on the raw scale."""
    pp = np.clip(p + shift, 1e-6, 1)
    q = np.clip(1 - 1 / (2 * pp), alphas[0], alphas[-1])
    Q = np.sort(Q, 1)
    # vectorised linear interpolation
    j = np.clip(np.searchsorted(alphas, q) - 1, 0, len(alphas) - 2)
    a0, a1 = alphas[j], alphas[j + 1]
    r = np.arange(len(q))
    v = Q[r, j] + (q - a0) / (a1 - a0) * (Q[r, j + 1] - Q[r, j])
    return np.where(pp > 0.5, np.clip(kap * v, 0, cap), 0.0)


def extra_feats(X, Xt):
    X, Xt = X.copy(), Xt.copy()
    for Z in (X, Xt):
        lp, lh = Z["log_prev_month_spending"], Z["log_historical_spending"]
        Z["lprev_minus_lhist"] = lp - lh
        Z["l_atv_x_freq"] = np.log1p(Z["avg_transaction_value"].clip(lower=0) * Z["purchase_frequency"])
        Z["l_hist_per_tx"] = np.log1p(Z["historical_spending"] / (Z["total_transactions"] + 1))
        Z["l_hist_per_age"] = np.log1p(Z["hist_per_day"])
    return X, Xt


def prev_offset(Z):
    """log prev_month_spending, fallback log hist - 1.05, then 6.5 (feature-only, no target)."""
    lp = Z["log_prev_month_spending"].values
    o = np.where(np.isnan(lp), Z["log_historical_spending"].values - 1.05, lp)
    return np.where(np.isnan(o), 6.5, o)

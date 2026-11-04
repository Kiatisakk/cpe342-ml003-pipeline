# XGBoost multi-quantile model of the positive part (log scale), optional offset target
# r = log y - off(x), off = log prev_month_spending (feature only), plus hurdle median with a saved classifier p.
# usage: python t123/research/t3/exp_xgbq.py NAME [--offset none|prev|lin] [--p lgbh_log] [--lr 0.05] [--depth 7] [--seed 0]
import sys, os, time, argparse, numpy as np, pandas as pd, xgboost as xgb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load, nmae, fold_scores, save_oof, hurdle_median, NT
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

ap = argparse.ArgumentParser()
ap.add_argument("name"); ap.add_argument("--offset", default="none"); ap.add_argument("--p", default="lgbh_log")
ap.add_argument("--lr", type=float, default=0.05); ap.add_argument("--depth", type=int, default=7)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--mcw", type=float, default=5)
ap.add_argument("--extra", action="store_true", help="add log-ratio features")
ap.add_argument("--wscale", type=float, default=0.0, help="feature-only weight exp(off)^wscale")
a = ap.parse_args()
ALPHAS = np.array([0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
OD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "oof")
D = load(); X, Xt, y, F = D["X"].copy(), D["Xt"].copy(), D["y"], D["folds"]
n, nt = len(y), len(Xt)
pp = np.load(os.path.join(OD, f"{a.p}_p.npy")); Op, Tp = pp[:n], pp[n:]

if a.extra:
    for Z in (X, Xt):
        lp, lh = Z["log_prev_month_spending"], Z["log_historical_spending"]
        Z["lprev_minus_lhist"] = lp - lh
        Z["l_atv_x_freq"] = np.log1p(Z["avg_transaction_value"].clip(lower=0) * Z["purchase_frequency"])
        Z["l_hist_per_tx"] = np.log1p(Z["historical_spending"] / (Z["total_transactions"] + 1))
        Z["l_hist_per_age"] = np.log1p(Z["hist_per_day"])

def offset(Z, lin=None):
    if a.offset == "none": return np.zeros(len(Z))
    lp = Z["log_prev_month_spending"].values
    if a.offset == "prev":
        fb = Z["log_historical_spending"].values - 1.05     # median log(prev/hist) ~ -1.05 (fallback only)
        o = np.where(np.isnan(lp), fb, lp); return np.where(np.isnan(o), 6.5, o)
    return lin(Z)

t0 = time.time(); OQ = np.zeros((n, len(ALPHAS))); TQ = np.zeros((nt, len(ALPHAS))); its = []
for k, (tr, va) in enumerate(F):
    Xa, ya = X.iloc[tr], y[tr]; pos = ya > 0
    Xp, lt = Xa[pos], np.log(ya[pos])
    lin = None
    if a.offset == "lin":   # linear model of log y on log spend features + segment dummies, fit on train-fold payers
        cols = ["log_prev_month_spending", "log_historical_spending", "log_avg_transaction_value"]
        def design(Z):
            M = Z[cols].copy(); M = M.fillna(M.median() if True else 0)
            for s in range(4): M[f"s{s}"] = (Z["segment"].values == s).astype(float)
            return M.values
        med = Xp[cols].median()
        def design(Z, med=med):
            M = Z[cols].fillna(med).values
            S = np.column_stack([(Z["segment"].values == s) for s in range(4)]).astype(float)
            return np.column_stack([M, S, M * S[:, [3]]])
        lr_ = LinearRegression().fit(design(Xp), lt); lin = lambda Z: lr_.predict(design(Z))
    if a.offset == "linfull":  # per-segment ridge on all features (median-filled, + missing flags), train-fold payers only
        from sklearn.linear_model import Ridge
        cols = list(Xp.columns); med = Xp[cols].median(); mu_, sd_ = None, None
        def dz(Z):
            M = Z[cols].fillna(med).values; return np.column_stack([M, Z[cols].isna().values[:, :32].astype(float)])
        Mp = dz(Xp); mu_, sd_ = Mp.mean(0), Mp.std(0) + 1e-9
        segs = Xp["segment"].values.astype(int); RM = {}
        for s in range(4): RM[s] = Ridge(1.0).fit((Mp[segs == s] - mu_) / sd_, lt[segs == s])
        def lin(Z):
            M = (dz(Z) - mu_) / sd_; sg = Z["segment"].values.astype(int); o = np.zeros(len(Z))
            for s in range(4): o[sg == s] = RM[s].predict(M[sg == s])
            return o
    oa = offset(Xp, lin); r = lt - oa
    w = np.exp(oa * a.wscale) if a.wscale else np.ones(len(r)); w = w / w.mean()
    i, j = train_test_split(np.arange(len(r)), test_size=0.1, random_state=a.seed)
    m = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=ALPHAS, n_estimators=20000, learning_rate=a.lr,
                         early_stopping_rounds=150, max_depth=a.depth, min_child_weight=a.mcw, subsample=0.8, colsample_bytree=0.7,
                         reg_lambda=2.0, tree_method="hist", n_jobs=NT, random_state=a.seed)
    m.fit(Xp.iloc[i], r[i], sample_weight=w[i], eval_set=[(Xp.iloc[j], r[j])], sample_weight_eval_set=[w[j]], verbose=False)
    its.append(m.best_iteration)
    OQ[va] = np.exp(np.sort(m.predict(X.iloc[va]), 1) + offset(X.iloc[va], lin)[:, None])
    TQ += np.exp(np.sort(m.predict(Xt), 1) + offset(Xt, lin)[:, None]) / 5
    print(f"fold {k} {time.time()-t0:.0f}s it {its[-1]}", flush=True)
o = hurdle_median(Op, OQ, ALPHAS); t = hurdle_median(Tp, TQ, ALPHAS)
save_oof(a.name, o, t); np.save(os.path.join(OD, f"{a.name}_Q.npy"), np.r_[OQ, TQ])
mm, s = fold_scores(y, o, F)
print(f"RESULT {a.name}: OOF {nmae(y, o):.5f} fold mean {mm:.5f} sd {s:.5f} | iters {its}")

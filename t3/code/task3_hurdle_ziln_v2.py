# Task 3 v2 — Hurdle GBM + ZILN (NMAE)
# v2: (1) exact imputation from identities  hist = tx*avg, pod = tx*disc
#     (2) segment imputation (classifier trained on TRAIN rows only) + flag
#     (3) mu / q: 15000 trees (v1 hit the 4000 cap); clf / sig: 4000
#     (4) whale expert: mu model trained on segment-3 payers, mixed by P(seg3)
# lr 0.003, early stop 100 on inner 10% holdout. NO SUBMIT - writes /kaggle/working/preds only
import os, time, json, warnings, numpy as np, pandas as pd, xgboost as xgb
from scipy.stats import norm
from sklearn.model_selection import StratifiedKFold, train_test_split
warnings.filterwarnings("ignore")
SEED, NF, LR, ES = 42, 5, 0.003, 100
TREES = {"clf": 4000, "sig": 4000, "mu": 15000, "q": 15000, "mae": 4000, "whale": 15000}
SMOKE = os.environ.get("SMOKE") == "1"
if SMOKE: TREES = {k: 60 for k in TREES}; LR = 0.1
DEVICE = "cuda" if os.system("nvidia-smi > /dev/null 2>&1") == 0 else "cpu"
ROOT = os.environ.get("DATA_ROOT", "/kaggle/input"); OUT = os.environ.get("OUT_DIR", "/kaggle/working/preds")
print("xgboost", xgb.__version__, "| device", DEVICE, "| trees", TREES, flush=True)

def walk_find(name, must=None):
    for d, _, fs in os.walk(ROOT, followlinks=True):
        if name in fs and (must is None or must in d): return os.path.join(d, name)
for _try in range(30):                                   # input mount can lag a few seconds
    TRAIN, TEST = walk_find("train.csv", "03_regression"), walk_find("test.csv", "03_regression")
    if TRAIN and TEST: break
    time.sleep(10)
if not (TRAIN and TEST):
    print("input tree:", [(d, fs[:3]) for d, _, fs in os.walk(ROOT)][:30], flush=True)
SAMPLE = walk_find("sample_submission.csv")
assert TRAIN and TEST, "data not found"
tr, te = pd.read_csv(TRAIN), pd.read_csv(TEST)
if SMOKE: tr = tr.sample(8000, random_state=0).reset_index(drop=True)
TARGET = "spending_30d"; y = tr[TARGET].values.astype(float)
ids = pd.read_csv(SAMPLE)["id"].values if SAMPLE else np.array([f"ANS{i+1:05d}" for i in range(len(te))])
assert len(ids) == len(te)
print(tr.shape, te.shape, flush=True)
def nmae(yt, yp): return 1 / (1 + np.abs(yt - yp).sum() / yt.sum())

# ---------- 1. exact imputation ----------
def exact_fill(d):
    d = d.copy(); filled = 0
    for _ in range(3):
        h, tx, av = d["historical_spending"], d["total_transactions"], d["avg_transaction_value"]
        po, dr = d["purchases_on_discount"], d["discount_rate_used"]
        rules = [
            ("historical_spending",   h.isna() & tx.notna() & av.notna(), tx * av),
            ("total_transactions",    tx.isna() & h.notna() & av.notna() & (av > 0), h / av),
            ("avg_transaction_value", av.isna() & h.notna() & tx.notna() & (tx > 0), h / tx),
            ("purchases_on_discount", po.isna() & tx.notna() & dr.notna(), tx * dr),
            ("total_transactions",    tx.isna() & po.notna() & dr.notna() & (dr > 0), po / dr),
            ("discount_rate_used",    dr.isna() & po.notna() & tx.notna() & (tx > 0), po / tx),
        ]
        for col, m, val in rules:
            m = m & d[col].isna(); d.loc[m, col] = val[m]; filled += int(m.sum())
    return d, filled
COLS5 = ["historical_spending", "total_transactions", "avg_transaction_value", "purchases_on_discount", "discount_rate_used"]
na_before = tr[COLS5].isna().mean().round(4).to_dict()
tr, f1 = exact_fill(tr); te, f2 = exact_fill(te)
print(f"exact fill: train {f1} cells, test {f2} cells | NaN before {na_before} -> after {tr[COLS5].isna().mean().round(4).to_dict()}", flush=True)

# ---------- 2. features ----------
SPEND = ["historical_spending", "prev_month_spending", "avg_transaction_value"]
def add_feats(d):
    d = d.copy(); e = 1.0
    d["hist_per_day"]      = d["historical_spending"] / (d["account_age_days"] + e)
    d["tx_per_day"]        = d["total_transactions"] / (d["account_age_days"] + e)
    d["exp_monthly_spend"] = d["avg_transaction_value"] * d["purchase_frequency"]
    d["prev_vs_hist"]      = d["prev_month_spending"] / (d["historical_spending"] + e)
    d["prev_x_freq"]       = d["prev_month_spending"] * d["purchase_frequency"]
    d["recency_ratio"]     = d["days_since_last_purchase"] / (d["days_since_last_login"] + e)
    d["play_per_session"]  = d["total_playtime_hours"] / (d["sessions_per_week"] + e)
    for c in SPEND: d[f"log_{c}"] = np.log1p(d[c].clip(lower=0))
    d["n_missing"] = d.isna().sum(axis=1)
    return d
X, Xt = add_feats(tr.drop(columns=[TARGET])), add_feats(te)
FEATS = [c for c in Xt.columns if c != "player_id"]
X, Xt = X[FEATS].copy(), Xt[FEATS].copy()

# ---------- 3. segment imputation (fit on TRAIN rows with known segment only) ----------
SF = [c for c in FEATS if c != "segment"]
known = X["segment"].notna().values
sc = xgb.XGBClassifier(objective="multi:softprob", n_estimators=TREES["clf"], learning_rate=max(LR, 0.02), early_stopping_rounds=ES,
                       max_depth=6, subsample=0.8, colsample_bytree=0.7, tree_method="hist", device=DEVICE, random_state=SEED)
i_a, i_b = train_test_split(np.where(known)[0], test_size=0.1, random_state=SEED, stratify=X["segment"].values[known])
sc.fit(X.iloc[i_a][SF], X["segment"].values[i_a].astype(int), eval_set=[(X.iloc[i_b][SF], X["segment"].values[i_b].astype(int))], verbose=False)
print("segment imputer holdout acc:", round(float((sc.predict(X.iloc[i_b][SF]) == X["segment"].values[i_b]).mean()), 4), flush=True)
for D in (X, Xt):
    miss = D["segment"].isna().values
    D["seg_imputed"] = miss.astype(int)
    if miss.any():
        P = sc.predict_proba(D.loc[miss, SF])
        D.loc[miss, "segment"] = P.argmax(1)
        D["p_whale"] = np.where(D["segment"].values == 3, 1.0, 0.0); D.loc[miss, "p_whale"] = P[:, 3]
    else:
        D["p_whale"] = (D["segment"].values == 3).astype(float)
FEATS = list(X.columns)
print(len(FEATS), "features", flush=True)

# ---------- 4. models ----------
def base(n, **kw):
    return {**dict(n_estimators=n, learning_rate=LR, early_stopping_rounds=ES, max_depth=7, min_child_weight=5, subsample=0.8,
                   colsample_bytree=0.7, reg_lambda=2.0, tree_method="hist", device=DEVICE, random_state=SEED), **kw}
ALPHAS = np.array([0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6])
def fit_es(model, Xa, ya, strat=None):
    i_tr, i_es = train_test_split(np.arange(len(ya)), test_size=0.1, random_state=SEED, stratify=strat)
    model.fit(Xa.iloc[i_tr], ya[i_tr], eval_set=[(Xa.iloc[i_es], ya[i_es])], verbose=False)
    return model

pos_all = y > 0
strat = np.where(pos_all, 1 + pd.qcut(pd.Series(y).where(pos_all).rank(method="first"), 4, labels=False).fillna(-1).values, 0).astype(int)
folds = list(StratifiedKFold(NF, shuffle=True, random_state=SEED).split(X, strat))
n, nt = len(y), len(Xt)
KEYS = ["mae", "p", "mu", "sig", "muw"]
O = {k: np.zeros(n) for k in KEYS}; O["q"] = np.zeros((n, len(ALPHAS)))
T = {k: np.zeros(nt) for k in KEYS}; T["q"] = np.zeros((nt, len(ALPHAS)))
best_it = {}
def log_it(k, m): best_it.setdefault(k, []).append(int(m.best_iteration))

t_all = time.time()
for k, (a, b) in enumerate(folds):
    t0 = time.time(); Xa, Xb, ya = X.iloc[a], X.iloc[b], y[a]; pa = ya > 0
    m = fit_es(xgb.XGBRegressor(objective="reg:absoluteerror", **base(TREES["mae"], learning_rate=max(LR, 0.02))), Xa, ya, strat[a])
    O["mae"][b] = m.predict(Xb); T["mae"] += m.predict(Xt) / NF; log_it("mae", m)
    c = fit_es(xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", **base(TREES["clf"])), Xa, pa.astype(int), pa.astype(int))
    O["p"][b] = c.predict_proba(Xb)[:, 1]; T["p"] += c.predict_proba(Xt)[:, 1] / NF; log_it("clf", c)
    Xp, lp = Xa[pa], np.log1p(ya[pa])
    r = fit_es(xgb.XGBRegressor(objective="reg:squarederror", **base(TREES["mu"])), Xp, lp)
    O["mu"][b] = r.predict(Xb); T["mu"] += r.predict(Xt) / NF; log_it("mu", r)
    res2 = (lp - r.predict(Xp)) ** 2
    s = fit_es(xgb.XGBRegressor(objective="reg:squarederror", **base(TREES["sig"], max_depth=5)), Xp, res2)
    sg = lambda Z: np.sqrt(np.clip(s.predict(Z), 1e-4, None))
    O["sig"][b] = sg(Xb); T["sig"] += sg(Xt) / NF; log_it("sig", s)
    # whale expert: payers with segment==3 (observed or imputed)
    wm = (Xp["segment"].values == 3)
    w = fit_es(xgb.XGBRegressor(objective="reg:squarederror", **base(TREES["whale"], max_depth=6)), Xp[wm], lp[wm])
    O["muw"][b] = w.predict(Xb); T["muw"] += w.predict(Xt) / NF; log_it("whale", w)
    qm = fit_es(xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=ALPHAS, **base(TREES["q"])), Xp, lp)
    O["q"][b] = np.sort(qm.predict(Xb), 1); T["q"] += np.sort(qm.predict(Xt), 1) / NF; log_it("q", qm)
    print(f"fold {k}: {time.time()-t0:.0f}s | best_iter " + ", ".join(f"{kk}={v[-1]}" for kk, v in best_it.items()), flush=True)
print(f"total {time.time()-t_all:.0f}s")
for kk, v in best_it.items():
    if max(v) >= TREES[kk if kk in TREES else "clf"] - ES - 1: print(f"WARNING: {kk} used ~all {TREES[kk]} trees")

# ---------- 5. distribution -> MAE-optimal ----------
def mix_mu(mu, muw, pw, g): return mu + g * pw * (muw - mu)          # g=0: general, g=1: full expert for whales
def pred_C(p, mu, sig, scale=1.0, shift=0.0):
    q = np.clip(1 - 1 / (2 * np.clip(p + shift, 1e-6, 1)), 1e-4, 0.5)
    return np.where(p + shift > 0.5, np.clip(np.expm1(mu + scale * sig * norm.ppf(q)), 0, 300000), 0.0)
def pred_D(p, Q, shift=0.0):
    q = np.clip(1 - 1 / (2 * np.clip(p + shift, 1e-6, 1)), ALPHAS[0], ALPHAS[-1])
    v = np.array([np.interp(qi, ALPHAS, Qi) for qi, Qi in zip(q, Q)])
    return np.where(p + shift > 0.5, np.clip(np.expm1(v), 0, 300000), 0.0)
pw_tr, pw_te = X["p_whale"].values, Xt["p_whale"].values

R = {"zero": nmae(y, np.zeros(n)), "A_mae": nmae(y, np.clip(O["mae"], 0, None)),
     "B_median": nmae(y, np.where(O["p"] > .5, np.expm1(O["mu"]), 0)),
     "C_theory": nmae(y, pred_C(O["p"], O["mu"], O["sig"])), "D_theory": nmae(y, pred_D(O["p"], O["q"]))}
gC = max(((sc_, sh) for sc_ in np.arange(0.6, 2.61, 0.1) for sh in np.arange(-0.1, 0.101, 0.025)),
         key=lambda t: nmae(y, pred_C(O["p"], O["mu"], O["sig"], *t)))
R["C_tuned"] = nmae(y, pred_C(O["p"], O["mu"], O["sig"], *gC))
gW = max(((g, sc_) for g in np.arange(0, 1.01, 0.1) for sc_ in np.arange(0.6, 2.61, 0.1)),
         key=lambda t: nmae(y, pred_C(O["p"], mix_mu(O["mu"], O["muw"], pw_tr, t[0]), O["sig"], t[1], gC[1])))
R["W_whale_expert"] = nmae(y, pred_C(O["p"], mix_mu(O["mu"], O["muw"], pw_tr, gW[0]), O["sig"], gW[1], gC[1]))
gD = max(np.arange(-0.1, 0.101, 0.025), key=lambda sh: nmae(y, pred_D(O["p"], O["q"], sh)))
R["D_tuned"] = nmae(y, pred_D(O["p"], O["q"], gD))
print("C params", np.round(gC, 3), "| whale (g, scale)", np.round(gW, 2), "| D shift", round(float(gD), 3))

PA = np.clip(O["mae"], 0, None); PC = pred_C(O["p"], O["mu"], O["sig"], *gC)
PW = pred_C(O["p"], mix_mu(O["mu"], O["muw"], pw_tr, gW[0]), O["sig"], gW[1], gC[1]); PD = pred_D(O["p"], O["q"], gD)
grid = [(wa, wc, ww, round(1 - wa - wc - ww, 2)) for wa in np.arange(0, 1.01, .1) for wc in np.arange(0, 1.01 - wa, .1)
        for ww in np.arange(0, 1.01 - wa - wc, .1)]
wb = max(grid, key=lambda w: nmae(y, w[0]*PA + w[1]*PC + w[2]*PW + w[3]*PD))
R["E_blend"] = nmae(y, wb[0]*PA + wb[1]*PC + wb[2]*PW + wb[3]*PD)
print("blend (A, C, W, D):", np.round(wb, 2))
print("v1 reference: C_ziln_tuned 0.7893")
for kk, v in R.items(): print(f"{kk:16s} {v:.4f}  (err ratio {1/v-1:.4f})")
best_oof = wb[0]*PA + wb[1]*PC + wb[2]*PW + wb[3]*PD
seg = tr["segment"].fillna(-1).values
print(pd.DataFrame({"seg": seg, "err": np.abs(y - best_oof), "y": y}).groupby("seg").agg(n=("y", "size"), sum_err=("err", "sum"), sum_y=("y", "sum"))
      .assign(err_ratio=lambda d: (d.sum_err / d.sum_y).round(3), share_err=lambda d: (d.sum_err / d.sum_err.sum()).round(3)))

# ---------- 6. test + save ----------
TA = np.clip(T["mae"], 0, None); TC = pred_C(T["p"], T["mu"], T["sig"], *gC)
TW = pred_C(T["p"], mix_mu(T["mu"], T["muw"], pw_te, gW[0]), T["sig"], gW[1], gC[1]); TD = pred_D(T["p"], T["q"], gD)
choices = {"A_mae": TA, "C_tuned": TC, "W_whale_expert": TW, "D_tuned": TD, "E_blend": wb[0]*TA + wb[1]*TC + wb[2]*TW + wb[3]*TD}
best_name = max(choices, key=lambda kk: R[kk]); pred = np.round(choices[best_name], 2)
os.makedirs(OUT, exist_ok=True)
pd.DataFrame({"id": ids, "pred": pred}).to_csv(f"{OUT}/t3_ziln_v2.csv", index=False)
for kk, v in choices.items(): pd.DataFrame({"id": ids, "pred": np.round(v, 2)}).to_csv(f"{OUT}/t3_v2_{kk}.csv", index=False)
pd.DataFrame({"y": y, "A": PA, "C": PC, "W": PW, "D": PD, "p": O["p"], "mu": O["mu"], "sig": O["sig"], "muw": O["muw"]}).to_csv(f"{OUT}/t3_v2_oof.csv", index=False)
json.dump({"oof_nmae": {kk: round(float(v), 5) for kk, v in R.items()}, "best": best_name, "blend_w": [float(x) for x in wb],
           "C": [float(x) for x in gC], "W": [float(x) for x in gW], "D_shift": float(gD), "best_iter": best_it},
          open(f"{OUT}/t3_ziln_v2_summary.json", "w"), indent=1)
print("USE:", best_name, "| OOF NMAE", round(R[best_name], 4), "| test zero%", round((pred == 0).mean(), 3), "mean", round(pred.mean(), 1))
print("wrote", OUT, "- nothing submitted")

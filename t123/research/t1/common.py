"""Shared harness for T1 research: data loading, feature engineering, CV, metrics.

Rules respected: test.csv is only ever *transformed* with statistics fitted on train rows
(labeled + unlabeled train). Nothing is fitted on test.
"""
import os, json, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, fbeta_score

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "..", "..", "..", "cpe342-kaggle-main", "01_binary_classification")
OUT = os.path.join(ROOT, "out")
os.makedirs(OUT, exist_ok=True)
TARGET = "is_cheater"
SIGNAL = ["kill_death_ratio", "headshot_percentage", "win_rate", "accuracy_score", "kill_consistency",
          "reaction_time_ms", "account_age_days", "level", "level_progression_speed", "friend_network_size",
          "reports_received", "aiming_smoothness", "spray_control_score", "game_sense_score", "crosshair_placement"]


def load():
    """Returns labeled train (file order), unlabeled train, test, test ids."""
    tr = pd.read_csv(os.path.join(DATA, "train.csv"))
    te = pd.read_csv(os.path.join(DATA, "test.csv"))
    lab = tr[tr[TARGET].notna()].reset_index(drop=True)
    unl = tr[tr[TARGET].isna()].reset_index(drop=True)
    ids = np.array([f"ANS{i+1:05d}" for i in range(len(te))])
    return lab, unl, te, ids


ALL_RAW = None


def raw_cols(df):
    return [c for c in df.columns if c not in ("player_id", TARGET)]


# ------------------------------------------------------------------ features
def fe_baseline(df, ref, drop_noise=True):
    """Re-implementation of task1_gbm_pipeline.FeatureEngineer (numeric version; the categorical
    bins are encoded as integer codes). `ref` = frame the statistics are fitted on (train only)."""
    cols = SIGNAL if drop_noise else raw_cols(df)
    X = df[cols].copy()
    lv, ag, lp = df["level"], df["account_age_days"], df["level_progression_speed"]
    for c in ["reports_received", "friend_network_size", "account_age_days", "crosshair_placement", "level"]:
        X[f"na_{c}"] = df[c].isna().astype(np.int8)
    X["n_missing"] = df[SIGNAL].isna().sum(axis=1)
    X["age_lo"] = lv * 0.15 / lp; X["age_hi"] = lv * 1.08 / lp
    X["age_filled"] = ag.fillna(np.sqrt(X["age_lo"] * X["age_hi"]))
    X["lv_lo"] = lp * ag / 1.08; X["lv_hi"] = lp * ag / 0.15
    X["lvl_filled"] = lv.fillna(np.sqrt(X["lv_lo"] * X["lv_hi"]))
    X["r_hat"] = lp * ag / lv
    rep, ch, age, fr = df["reports_received"], df["crosshair_placement"], X["age_filled"], df["friend_network_size"]
    X["zone_cheat"] = ((rep == 4) | (ch < 0.82) | (X["lvl_filled"] < 28)).astype(np.int8)
    X["zone_legit"] = ((rep >= 20) | (ch > 0.94) | (X["lvl_filled"] > 96) | (age > 1000) | (age < 15) | (fr < 10) | (fr > 269)).astype(np.int8)
    for c in ["account_age_days", "friend_network_size", "level_progression_speed", "reports_received"]:
        X[f"log_{c}"] = np.log1p(df[c].clip(lower=0))
    X["kd_x_hs"] = df["kill_death_ratio"] * df["headshot_percentage"]
    X["acc_x_cross"] = df["accuracy_score"] * df["crosshair_placement"]
    X["reports_per_100d"] = 100 * rep / (age + 1)
    X["friends_per_100d"] = 100 * fr / (age + 1)
    X["level_per_100d"] = 100 * X["lvl_filled"] / (age + 1)
    sk = ["kill_death_ratio", "headshot_percentage", "accuracy_score", "win_rate", "kill_consistency",
          "aiming_smoothness", "spray_control_score", "game_sense_score", "crosshair_placement"]
    mu, sd = ref[sk].median(), ref[sk].std() + 1e-9
    Z = (df[sk] - mu) / sd
    X["skill_z_sum"] = Z.sum(axis=1, min_count=3); X["skill_z_std"] = Z.std(axis=1)
    X["freq_reports"] = rep.map(ref["reports_received"].value_counts(normalize=True)).astype(float)
    X["freq_friends"] = fr.map(ref["friend_network_size"].value_counts(normalize=True)).astype(float)
    X["age_segment"] = pd.cut(age, [-np.inf, 15, 100, 200, 1000, np.inf], labels=False).astype(float)
    return X.replace([np.inf, -np.inf], np.nan)


# ------------------------------------------------------------------ metrics
def f2m(y, pred):
    return fbeta_score(y, pred, beta=2, average="macro")


def f2b(y, pred):
    return fbeta_score(y, pred, beta=2)


def best_thr(y, p, kind="macro", lo=0.20, hi=0.85, n=261):
    """Threshold search on quantiles of p. Returns (thr, score, pos_rate)."""
    fn = f2m if kind == "macro" else f2b
    if kind != "macro": lo, hi = 0.25, 0.95
    qs = np.linspace(lo, hi, n)
    grid = np.quantile(p, 1 - qs)  # predict top-q as positive
    # fast vectorised F-beta via sorting
    order = np.argsort(-p); ys = y[order]; P = len(y); npos = ys.sum()
    tp_c = np.cumsum(ys)
    best = (-1, None, None)
    for t in grid:
        k = int((p >= t).sum())
        tp = tp_c[k - 1] if k > 0 else 0; fp = k - tp; fn_ = npos - tp; tn = P - k - fn_
        b2 = 4.0
        f_pos = (1 + b2) * tp / ((1 + b2) * tp + b2 * fn_ + fp) if tp > 0 else 0.0
        if kind == "macro":
            f_neg = (1 + b2) * tn / ((1 + b2) * tn + b2 * fp + fn_) if tn > 0 else 0.0
            s = 0.5 * (f_pos + f_neg)
        else:
            s = f_pos
        if s > best[0]: best = (s, t, k / P)
    return best[1], best[0], best[2]


def nested_thr_score(y, p, folds, kind="macro"):
    """Threshold picked as a *positive-rate* on the other folds' OOF, applied to the held-out fold."""
    pred = np.zeros(len(y), int)
    for a, b in folds:
        _, _, rate = best_thr(y[a], p[a], kind)
        t = np.quantile(p[a], 1 - rate)
        pred[b] = (p[b] >= t).astype(int)
    return (f2m if kind == "macro" else f2b)(y, pred)


def report(name, y, p, seed=42, extra=None, save=True):
    folds = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(p, y))
    _, sm, rm = best_thr(y, p, "macro")
    _, sb, rb = best_thr(y, p, "binary")
    r = dict(name=name, auc=round(roc_auc_score(y, p), 5), f2m=round(sm, 5), f2m_nested=round(nested_thr_score(y, p, folds), 5),
             f2b=round(sb, 5), f2b_nested=round(nested_thr_score(y, p, folds, "binary"), 5), pos_rate_m=round(rm, 4), pos_rate_b=round(rb, 4))
    if extra: r.update(extra)
    print(json.dumps(r), flush=True)
    if save:
        with open(os.path.join(OUT, "results.jsonl"), "a") as f: f.write(json.dumps(r) + "\n")
    return r


def folds_for(y, seed):
    return list(StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))


# ------------------------------------------------------------------ CV runners
LGB_BASE = dict(num_leaves=39, min_child_samples=8, subsample=0.814, subsample_freq=1, colsample_bytree=0.455,
                reg_lambda=4.01, reg_alpha=0.0045, min_split_gain=0.02, max_bin=511, verbose=-1, n_jobs=4)


def cv_lgb(X, y, seed=42, params=None, lr=0.05, Xt=None, rounds=5000, es=200, inner_seed_offset=0, fit_extra=None, ret_iters=False):
    """Same scheme as the baseline: 5-fold StratifiedKFold(seed); 10% inner split for early stopping."""
    import lightgbm as lgb
    p = dict(LGB_BASE); p.update(params or {})
    oof = np.zeros(len(y)); tp = np.zeros(len(Xt)) if Xt is not None else None; its = []
    for k, (a, b) in enumerate(folds_for(y, seed)):
        rng = np.random.default_rng(seed + k + inner_seed_offset); a = rng.permutation(a); cut = int(0.9 * len(a))
        m = lgb.LGBMClassifier(n_estimators=rounds, learning_rate=lr, random_state=seed + k, **p)
        m.fit(X.iloc[a[:cut]], y[a[:cut]], eval_set=[(X.iloc[a[cut:]], y[a[cut:]])], eval_metric="auc",
              callbacks=[lgb.early_stopping(es, verbose=False)])
        oof[b] = m.predict_proba(X.iloc[b])[:, 1]; its.append(m.best_iteration_)
        if Xt is not None: tp += m.predict_proba(Xt)[:, 1] / 5
    return (oof, tp, its) if ret_iters else (oof, tp)


class Timer:
    def __init__(self): self.t = time.time()
    def __call__(self): return f"{(time.time() - self.t) / 60:.1f}m"


SEEDS = (42, 7, 2024)


def multi_seed(name, y, make_oof, seeds=SEEDS, save=True, extra=None):
    """make_oof(seed) -> oof probs. Reports each seed (fold split = seed) and mean +- sd."""
    rs, oofs = [], []
    for s in seeds:
        T = Timer(); o = make_oof(s); oofs.append(o)
        rs.append(report(f"{name} [seed {s}]", y, o, seed=s, save=False, extra=dict(time=T())))
    agg = dict(name=name, seeds=list(seeds))
    for k in ["auc", "f2m", "f2m_nested", "f2b", "f2b_nested"]:
        v = np.array([r[k] for r in rs]); agg[k] = round(v.mean(), 5); agg[k + "_sd"] = round(v.std(ddof=1), 5) if len(v) > 1 else 0
    if extra: agg.update(extra)
    print("AGG", json.dumps(agg), flush=True)
    if save:
        with open(os.path.join(OUT, "results_ms.jsonl"), "a") as f: f.write(json.dumps(agg) + "\n")
    return agg, oofs


# ------------------------------------------------------------------ v2: exact zone posteriors
AGE_ZONES = {"cheat_110_120": (110, 120), "legit_lt15": (0, 15), "legit_180_200": (180, 200), "legit_ge1000": (1000, 1e9),
             "young_lt110": (0, 110), "mid_120_180": (120, 180)}
LVL_ZONES = {"cheat_lt28": (0, 28), "legit_gt96": (96, 1e9)}


def _posterior_feats(obs_prior, grid, lik):
    """lik: (n, G) unnormalised likelihood on grid; obs_prior: (G,) prior weights. Returns posterior (n,G)."""
    w = lik * obs_prior[None, :]
    s = w.sum(1, keepdims=True)
    return np.where(s > 0, w / np.where(s > 0, s, 1), np.nan)


def fe_zone_post(df, ref):
    """Label-free posterior over age / level zones when age / level is missing, using
    lps = level * r / age with r ~ U(0.15, 1.08) and the empirical marginal prior fitted on `ref` (train rows only)."""
    out = pd.DataFrame(index=df.index)
    ag, lv, lp = df["account_age_days"].values, df["level"].values, df["level_progression_speed"].values
    # ---- age
    gA = np.arange(0.5, 1500, 1.0)
    pA = np.histogram(ref["account_age_days"].dropna(), np.arange(0, 1501, 1.0))[0].astype(float) + 0.5
    m = np.isnan(ag) & ~np.isnan(lv) & ~np.isnan(lp)
    E_age = np.full(len(df), np.nan)
    for nm in AGE_ZONES: out[f"pa_{nm}"] = np.nan
    if m.any():
        r = lp[m, None] * gA[None, :] / lv[m, None]
        lik = ((r >= 0.15) & (r <= 1.08)) * gA[None, :]
        P = _posterior_feats(pA, gA, lik)
        E_age[m] = (P * gA).sum(1)
        for nm, (a, b) in AGE_ZONES.items():
            out.loc[m, f"pa_{nm}"] = P[:, (gA >= a) & (gA < b)].sum(1)
    out["age_post_mean"] = np.where(np.isnan(ag), E_age, ag)
    for nm, (a, b) in AGE_ZONES.items():   # observed age -> exact indicator
        o = ~np.isnan(ag); out.loc[o, f"pa_{nm}"] = ((ag[o] >= a) & (ag[o] < b)).astype(float)
    # ---- level
    gL = np.arange(25.05, 100, 0.1)
    pL = np.histogram(ref["level"].dropna(), np.arange(25.0, 100.01, 0.1))[0].astype(float)[:len(gL)] + 0.5
    m = np.isnan(lv) & ~np.isnan(ag) & ~np.isnan(lp)
    E_l = np.full(len(df), np.nan)
    for nm in LVL_ZONES: out[f"pl_{nm}"] = np.nan
    if m.any():
        r = lp[m, None] * ag[m, None] / gL[None, :]
        lik = ((r >= 0.15) & (r <= 1.08)) / gL[None, :]
        P = _posterior_feats(pL, gL, lik)
        E_l[m] = (P * gL).sum(1)
        for nm, (a, b) in LVL_ZONES.items():
            out.loc[m, f"pl_{nm}"] = P[:, (gL >= a) & (gL < b)].sum(1)
    out["lvl_post_mean"] = np.where(np.isnan(lv), E_l, lv)
    for nm, (a, b) in LVL_ZONES.items():
        o = ~np.isnan(lv); out.loc[o, f"pl_{nm}"] = ((lv[o] >= a) & (lv[o] < b)).astype(float)
    out["p_cheat_zone"] = 1 - (1 - out["pa_cheat_110_120"].fillna(0)) * (1 - out["pl_cheat_lt28"].fillna(0))
    out["p_legit_zone"] = 1 - (1 - out[["pa_legit_lt15", "pa_legit_180_200", "pa_legit_ge1000"]].fillna(0).sum(1).clip(0, 1)) * (1 - out["pl_legit_gt96"].fillna(0))
    return out


def fe_v2(df, ref):
    return pd.concat([fe_baseline(df, ref), fe_zone_post(df, ref)], axis=1)


# ------------------------------------------------------------------ XGB / CatBoost runners (baseline-tuned params)
XGB_BASE = dict(max_depth=7, min_child_weight=1.41, subsample=0.998, colsample_bytree=0.696, reg_lambda=15.4, reg_alpha=11.8,
                gamma=0.0369, max_bin=128, grow_policy="lossguide", tree_method="hist", n_jobs=4)
CAT_BASE = dict(depth=5, l2_leaf_reg=0.683, random_strength=0.0706, border_count=254, bootstrap_type="Bernoulli", subsample=0.81, thread_count=4)


def cv_generic(fit_pred, X, y, seed=42, Xt=None, inner_seed_offset=0):
    oof = np.zeros(len(y)); tp = np.zeros(len(Xt)) if Xt is not None else None
    for k, (a, b) in enumerate(folds_for(y, seed)):
        rng = np.random.default_rng(seed + k + inner_seed_offset); a = rng.permutation(a); cut = int(0.9 * len(a))
        pv, pt = fit_pred(X.iloc[a[:cut]], y[a[:cut]], X.iloc[a[cut:]], y[a[cut:]], X.iloc[b], Xt, seed + k)
        oof[b] = pv
        if Xt is not None: tp += pt / 5
    return oof, tp


def xgb_fp(params=None, lr=0.05):
    import xgboost as xgb
    def f(Xa, ya, Xv, yv, Xb, Xt, rs):
        p = dict(XGB_BASE); p.update(params or {})
        m = xgb.XGBClassifier(n_estimators=5000, learning_rate=lr, early_stopping_rounds=200, eval_metric="auc", random_state=rs, **p)
        m.fit(Xa, ya, eval_set=[(Xv, yv)], verbose=False)
        return m.predict_proba(Xb)[:, 1], (m.predict_proba(Xt)[:, 1] if Xt is not None else None)
    return f


def cat_fp(params=None, lr=0.05):
    import catboost as cb
    def f(Xa, ya, Xv, yv, Xb, Xt, rs):
        p = dict(CAT_BASE); p.update(params or {})
        m = cb.CatBoostClassifier(iterations=5000, learning_rate=lr, eval_metric="AUC", od_type="Iter", od_wait=200, random_seed=rs, verbose=False, **p)
        m.fit(Xa, ya, eval_set=(Xv, yv), use_best_model=True)
        return m.predict_proba(Xb)[:, 1], (m.predict_proba(Xt)[:, 1] if Xt is not None else None)
    return f


# ------------------------------------------------------------------ v3: label-free imputers for missing key features
# target feature -> (bin edges for a multiclass imputer, {zone name: (lo, hi)})
IMPUTE_SPECS = {
    "reports_received": (np.arange(3.5, 30.5, 1.0), {"rep4": (3.5, 4.5), "rep_ge20": (19.5, 99)}),
    "friend_network_size": (np.array([-1, 9.5, 30, 50, 70, 90, 120, 150, 180, 210, 240, 269.5, 400]), {"fr_lt10": (-1, 9.5), "fr_gt269": (269.5, 999)}),
    "crosshair_placement": (np.array([0, .82, .845, .87, .895, .92, .94, 1]), {"ch_lt082": (0, .82), "ch_gt094": (.94, 1)}),
    "account_age_days": (np.array([0, 15, 50, 110, 120, 150, 180, 200, 350, 600, 900, 1000, 1300, 1600]),
                         {"age_lt15": (0, 15), "age_110_120": (110, 120), "age_180_200": (180, 200), "age_ge1000": (1000, 9e9), "age_lt110": (0, 110)}),
    "level": (np.array([0, 28, 40, 55, 70, 85, 96, 101]), {"lv_lt28": (0, 28), "lv_gt96": (96, 999)}),
}
_IMPUTERS = {}


def fit_imputers(ref, use_noise=True, seed=0):
    """Label-free: each imputer predicts the bin of one feature from all other raw columns (signal + noise),
    trained on the `ref` rows where that feature is observed. ref = train rows only."""
    import lightgbm as lgb
    cols = raw_cols(ref) if use_noise else SIGNAL
    imps = {}
    for f, (edges, zones) in IMPUTE_SPECS.items():
        obs = ref[f].notna().values
        Xi = ref.loc[obs, [c for c in cols if c != f]]
        yi = np.clip(np.searchsorted(edges, ref.loc[obs, f].values, side="right") - 1, 0, len(edges) - 2)
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.08, num_leaves=31, min_child_samples=50, subsample=0.8, subsample_freq=1,
                               colsample_bytree=0.8, verbose=-1, n_jobs=4, random_state=seed)
        m.fit(Xi, yi)
        imps[f] = (m, [c for c in cols if c != f], edges, zones)
    return imps


def fe_impute(df, imps):
    out = pd.DataFrame(index=df.index)
    for f, (m, cols, edges, zones) in imps.items():
        P = np.zeros((len(df), len(edges) - 1)); P[:, m.classes_] = m.predict_proba(df[cols])
        mids = (edges[:-1] + edges[1:]) / 2
        mids[0] = max(mids[0], edges[1] - (edges[2] - edges[1]) / 2); mids[-1] = min(mids[-1], edges[-2] + (edges[-2] - edges[-3]) / 2)
        miss = df[f].isna().values; v = df[f].values
        out[f"imp_{f}"] = np.where(miss, P @ mids, v)
        for z, (lo, hi) in zones.items():
            sel = (mids > lo) & (mids < hi)
            out[f"pz_{z}"] = np.where(miss, P[:, sel].sum(1), ((v >= lo) & (v < hi)).astype(float))
    return out


def fe_v3(df, ref, imps):
    return pd.concat([fe_v2(df, ref), fe_impute(df, imps)], axis=1)

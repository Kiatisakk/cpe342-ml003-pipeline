"""Round 13 test builder: final_v2 recipe (fe_v2 + MCAR mask x4) + pseudo-labelled UNLABELED TRAIN rows
(teacher = in-fold baseline model). Writes test probabilities per variant. No test rows are used in training.
Usage: VARS=soft,hard LR=0.05 SEEDS=42,7,2024 python t1/research/round13_build_test.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import lightgbm as lgb
import common
common.DATA = os.path.join(common.ROOT, '..', '..', 'cpe342-kaggle-main', '01_binary_classification')

LR = float(os.environ.get("LR", 0.05)); NC = 4
SEEDS_ = [int(s) for s in os.environ.get("SEEDS", "42,7,2024").split(",")]
VARS = os.environ.get("VARS", "soft,hard").split(",")
lab, unl, te, ids = load(); y = lab[TARGET].astype(int).values
ref = pd.concat([lab, unl]); X = fe_v2(lab, ref); XU = fe_v2(unl, ref); Xt = fe_v2(te, ref)
miss = lab[SIGNAL].isna().mean().values


def mask(df, rng):
    d = df.copy(); M = rng.random((len(d), len(SIGNAL))) < miss[None, :]
    v = d[SIGNAL].values.copy(); v[M] = np.nan; d[SIGNAL] = v; return fe_v2(d, ref)


def fit(Xa, ya, wa, Xv, yv, seed):
    m = lgb.LGBMClassifier(n_estimators=20000, learning_rate=LR, random_state=seed, **LGB_BASE)
    m.fit(Xa, ya, sample_weight=wa, eval_set=[(Xv, yv)], eval_metric="auc", callbacks=[lgb.early_stopping(200, verbose=False)])
    return m


tp = {v: np.zeros(len(Xt)) for v in ["base"] + VARS}; n = 0
for s in SEEDS_:
    for k, (a, b) in enumerate(folds_for(y, s)):
        rng = np.random.default_rng(s + k); a = rng.permutation(a); cut = int(.9 * len(a)); tr, va = a[:cut], a[cut:]
        Xl = pd.concat([X.iloc[tr]] + [mask(lab.iloc[tr], rng) for _ in range(NC)], ignore_index=True)
        yl = np.tile(y[tr], NC + 1); wl = np.ones(len(yl))
        XUm = pd.concat([XU] + [mask(unl, rng) for _ in range(NC)], ignore_index=True)
        m0 = fit(Xl, yl, wl, X.iloc[va], y[va], s + k); tp["base"] += m0.predict_proba(Xt)[:, 1]
        pu = np.tile(m0.predict_proba(XU)[:, 1], NC + 1)
        if "hard" in VARS:
            sel = (pu >= .80) | (pu <= .05)
            m = fit(pd.concat([Xl, XUm[sel]], ignore_index=True), np.r_[yl, (pu[sel] >= .5).astype(int)], np.r_[wl, np.full(sel.sum(), .5)], X.iloc[va], y[va], s + k)
            tp["hard"] += m.predict_proba(Xt)[:, 1]
        for name, sw in [("soft", 1.0), ("soft_w05", .5)]:
            if name in VARS:
                m = fit(pd.concat([Xl, XUm, XUm], ignore_index=True), np.r_[yl, np.ones(len(pu), int), np.zeros(len(pu), int)],
                        np.r_[wl, sw * pu, sw * (1 - pu)], X.iloc[va], y[va], s + k)
                tp[name] += m.predict_proba(Xt)[:, 1]
        n += 1; print(f"seed {s} fold {k} done", flush=True)
for v in tp:
    np.save(os.path.join(OUT, f"r13_test_{v}.npy"), tp[v] / n)
print("saved", list(tp))

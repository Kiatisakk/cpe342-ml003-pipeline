"""Round 13: pseudo-labels on the 13,184 UNLABELED TRAIN rows (no test rows used, Rule 2 safe),
on top of the final_v2 recipe (fe_v2 + MCAR mask aug). Metric = BINARY F2 (the LB metric), nested cut,
plus the test-completeness-weighted binary F2 used in round10 (tracks the LB).
Teacher = the in-fold baseline model (fit on fold-train only) -> no leakage into the held-out fold.
Variants: base | hard (p>=HI ->1, p<=LO ->0, weight W) | soft (each U row twice, weights p and 1-p) | iter2 (soft, teacher=soft student).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import lightgbm as lgb
import common
common.DATA = os.path.join(common.ROOT, '..', '..', 'cpe342-kaggle-main', '01_binary_classification')

LR = float(os.environ.get("LR", 0.05)); NC = 4; SEEDS_ = [int(s) for s in os.environ.get("SEEDS", "42,7").split(",")]
lab, unl, te, ids = load(); y = lab[TARGET].astype(int).values
ref = pd.concat([lab, unl]); X = fe_v2(lab, ref); XU = fe_v2(unl, ref)
miss = lab[SIGNAL].isna().mean().values
key = ["account_age_days", "friend_network_size", "reports_received", "aiming_smoothness", "crosshair_placement"]
ct = lab[key].notna().all(axis=1).values; cv = te[key].notna().all(axis=1).mean()
W_TEST = np.where(ct, cv / ct.mean(), (1 - cv) / (1 - ct.mean()))


def mask(df, rng):
    d = df.copy(); M = rng.random((len(d), len(SIGNAL))) < miss[None, :]
    v = d[SIGNAL].values.copy(); v[M] = np.nan; d[SIGNAL] = v; return fe_v2(d, ref)


def fit(Xa, ya, wa, Xv, yv, seed):
    m = lgb.LGBMClassifier(n_estimators=20000, learning_rate=LR, random_state=seed, **LGB_BASE)
    m.fit(Xa, ya, sample_weight=wa, eval_set=[(Xv, yv)], eval_metric="auc", callbacks=[lgb.early_stopping(200, verbose=False)])
    return m


def f2b_w(p, yy, w):
    tp = (w * (p & (yy == 1))).sum(); fp = (w * (p & (yy == 0))).sum(); fn = (w * (~p & (yy == 1))).sum()
    return 5 * tp / (5 * tp + 4 * fn + fp)


def nested(p, folds, w=None):
    """cut = positive rate chosen on other folds (binary F2), applied to held-out fold."""
    w = np.ones(len(y)) if w is None else w; out = []
    grid = np.linspace(.45, .80, 71)
    for a, b in folds:
        sc = [f2b_w(p[a] >= np.quantile(p[a], 1 - r), y[a], w[a]) for r in grid]; r = grid[int(np.argmax(sc))]
        out.append(f2b_w(p[b] >= np.quantile(p[a], 1 - r), y[b], w[b]))
    return np.array(out)


VARS = ["base", "hard", "soft", "soft_w05", "iter2"]
res = {v: [] for v in VARS}
for s in SEEDS_:
    oof = {v: np.zeros(len(y)) for v in VARS}; folds = folds_for(y, s)
    for k, (a, b) in enumerate(folds):
        rng = np.random.default_rng(s + k); a = rng.permutation(a); cut = int(.9 * len(a)); tr, va = a[:cut], a[cut:]
        Xl = pd.concat([X.iloc[tr]] + [mask(lab.iloc[tr], rng) for _ in range(NC)], ignore_index=True)
        yl = np.tile(y[tr], NC + 1); wl = np.ones(len(yl))
        XUm = pd.concat([XU] + [mask(unl, rng) for _ in range(NC)], ignore_index=True); nU = len(XU)
        m0 = fit(Xl, yl, wl, X.iloc[va], y[va], s + k); oof["base"][b] = m0.predict_proba(X.iloc[b])[:, 1]
        pu = np.tile(m0.predict_proba(XU)[:, 1], NC + 1)
        # hard: confident only
        sel = (pu >= .80) | (pu <= .05)
        m = fit(pd.concat([Xl, XUm[sel]], ignore_index=True), np.r_[yl, (pu[sel] >= .5).astype(int)], np.r_[wl, np.full(sel.sum(), .5)], X.iloc[va], y[va], s + k)
        oof["hard"][b] = m.predict_proba(X.iloc[b])[:, 1]
        # soft: every U row as 1 (w=p) and 0 (w=1-p)
        for name, sw in [("soft", 1.0), ("soft_w05", .5)]:
            m = fit(pd.concat([Xl, XUm, XUm], ignore_index=True), np.r_[yl, np.ones(len(pu), int), np.zeros(len(pu), int)],
                    np.r_[wl, sw * pu, sw * (1 - pu)], X.iloc[va], y[va], s + k)
            oof[name][b] = m.predict_proba(X.iloc[b])[:, 1]
            if name == "soft": msoft = m
        pu2 = np.tile(msoft.predict_proba(XU)[:, 1], NC + 1)
        m = fit(pd.concat([Xl, XUm, XUm], ignore_index=True), np.r_[yl, np.ones(len(pu2), int), np.zeros(len(pu2), int)],
                np.r_[wl, pu2, 1 - pu2], X.iloc[va], y[va], s + k)
        oof["iter2"][b] = m.predict_proba(X.iloc[b])[:, 1]
        print(f"seed {s} fold {k} done", flush=True)
    for v in VARS:
        res[v].append((roc_auc_score(y, oof[v]), nested(oof[v], folds), nested(oof[v], folds, W_TEST)))
    np.save(os.path.join(OUT, f"r13_oof_seed{s}.npy"), np.array([oof[v] for v in VARS]))

print("\nvariant | AUC | nested binF2 | nested binF2 test-weighted | gain vs base (plain, tw) | folds better (tw)")
for v in VARS:
    auc = np.mean([r[0] for r in res[v]]); f = np.concatenate([r[1] for r in res[v]]); fw = np.concatenate([r[2] for r in res[v]])
    f0 = np.concatenate([r[1] for r in res["base"]]); fw0 = np.concatenate([r[2] for r in res["base"]])
    print(f"{v:9s} {auc:.5f} {f.mean():.5f} {fw.mean():.5f}  {f.mean()-f0.mean():+.5f} {fw.mean()-fw0.mean():+.5f} +-{(fw-fw0).std():.5f}  {(fw>fw0).sum()}/{len(fw)}")

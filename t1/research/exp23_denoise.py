"""Noise removal, all inside the outer fold's TRAINING rows (held-out fold untouched), LGB fe_v2 lr 0.05, 3 seeds.

Label noise (confident-learning style): inner 5-fold OOF (lr 0.1) on fold-train -> p(given label) per row.
  drop2 / drop5 / drop10 : remove the 2/5/10% rows with lowest p(given label), retrain
  cl_thr                 : remove rows whose label disagrees with a confident inner prediction (p(label) < 0.1)
  reweight               : keep all rows, weight = p(given label) ** 0.5
Feature noise (null importance): per outer fold, LGB gain importance of every feature vs the max importance of
  5 shuffled copies of random features (null); drop features below the null.
  nullimp                : drop sub-null features, retrain
usage: python exp23_denoise.py [variants...]
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import *
import lightgbm as lgb
lab, unl, te, ids = load(); y = lab[TARGET].astype(int).values
X = fe_v2(lab, pd.concat([lab, unl]))


def fit(Xa, ya, Xv, yv, rs, lr=0.05, w=None):
    m = lgb.LGBMClassifier(n_estimators=5000, learning_rate=lr, random_state=rs, **LGB_BASE)
    m.fit(Xa, ya, sample_weight=w, eval_set=[(Xv, yv)], eval_metric="auc", callbacks=[lgb.early_stopping(100 if lr > 0.05 else 200, verbose=False)])
    return m


def inner_oof(Xt, yt, rs):
    p = np.zeros(len(yt))
    for k, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=rs).split(Xt, yt)):
        r = np.random.default_rng(rs + k); a = r.permutation(a); c = int(0.9 * len(a))
        p[b] = fit(Xt.iloc[a[:c]], yt[a[:c]], Xt.iloc[a[c:]], yt[a[c:]], rs + k, lr=0.1).predict_proba(Xt.iloc[b])[:, 1]
    return p


def null_keep(Xt, yt, rs):
    rng = np.random.default_rng(rs); Z = Xt.copy()
    for j in range(5): Z[f"null{j}"] = rng.permutation(Xt.iloc[:, rng.integers(Xt.shape[1])].values)
    imps = []
    for k in range(3):
        m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, random_state=rs + k, importance_type="gain", **LGB_BASE).fit(Z, yt)
        imps.append(m.feature_importances_)
    imp = pd.Series(np.mean(imps, 0), index=Z.columns)
    thr = imp[[f"null{j}" for j in range(5)]].max()
    return [c for c in Xt.columns if imp[c] > thr], imp


VARS = ["drop2", "drop5", "drop10", "cl_thr", "reweight", "nullimp"]
want = sys.argv[1:] or VARS
res = {v: [] for v in want}; info = {v: [] for v in want}
for s in SEEDS:
    oof = {v: np.zeros(len(y)) for v in want}
    for k, (a, b) in enumerate(folds_for(y, s)):
        r = np.random.default_rng(s + k); a = r.permutation(a); c = int(0.9 * len(a)); tr, va = a[:c], a[c:]
        need_p = any(v != "nullimp" for v in want)
        if need_p:
            p = inner_oof(X.iloc[tr], y[tr], s * 100 + k); pl = np.where(y[tr] == 1, p, 1 - p)
        for v in want:
            if v.startswith("drop"):
                q = int(v[4:]) / 100; keep = pl > np.quantile(pl, q)
                m = fit(X.iloc[tr[keep]], y[tr[keep]], X.iloc[va], y[va], s + k); info[v].append(round(1 - keep.mean(), 4))
            elif v == "cl_thr":
                keep = pl >= 0.1; info[v].append(round(1 - keep.mean(), 4))
                m = fit(X.iloc[tr[keep]], y[tr[keep]], X.iloc[va], y[va], s + k)
            elif v == "reweight":
                m = fit(X.iloc[tr], y[tr], X.iloc[va], y[va], s + k, w=np.sqrt(np.clip(pl, 0.01, 1)))
            else:
                cols, imp = null_keep(X.iloc[tr], y[tr], s + k); info[v].append(X.shape[1] - len(cols))
                if k == 0 and s == SEEDS[0]: print("  dropped by null importance (fold0):", sorted(set(X.columns) - set(cols)), flush=True)
                m = fit(X.iloc[tr][cols], y[tr], X.iloc[va][cols], y[va], s + k)
                oof[v][b] = m.predict_proba(X.iloc[b][cols])[:, 1]; continue
            oof[v][b] = m.predict_proba(X.iloc[b])[:, 1]
    for v in want:
        f = folds_for(y, s)
        res[v].append((nested_thr_score(y, oof[v], f), nested_thr_score(y, oof[v], f, "binary"), roc_auc_score(y, oof[v])))
        np.save(f"{OUT}/exp23_{v}_s{s}.npy", oof[v])
    print("seed", s, {v: tuple(round(x, 5) for x in res[v][-1]) for v in want}, flush=True)
for v in want:
    A = np.array(res[v])
    r = dict(name=f"exp23 lgb fe_v2 {v}", f2m_nested=round(A[:, 0].mean(), 5), f2m_nested_sd=round(A[:, 0].std(ddof=1), 5),
             f2b_nested=round(A[:, 1].mean(), 5), auc=round(A[:, 2].mean(), 5), removed=f"{np.mean(info[v]):.4g}" if info[v] else "-")
    print("AGG", json.dumps(r), flush=True); open(f"{OUT}/results_ms.jsonl", "a").write(json.dumps(r) + "\n")

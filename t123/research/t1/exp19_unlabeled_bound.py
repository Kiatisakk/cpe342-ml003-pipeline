"""Upper bound for semi-supervised learning with the 13,184 unlabeled TRAIN rows.
Inside each outer fold's training set, 13,184 labeled rows are treated as 'unlabeled' (U), the rest is L.
  (a) L only                       (b1) L + U hard pseudo-labels (|p-0.5| > 0.35, weight 0.5)
  (b2) L + U soft pseudo-labels (each U row twice, weights p and 1-p)
  (c) L + U with TRUE labels  = the ceiling any semi-supervised method could reach
Held-out folds are scored as usual (LGB fe_v2 lr 0.05, nested threshold, 3 split seeds)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import *
import lightgbm as lgb
lab, unl, te, ids = load(); y = lab[TARGET].astype(int).values
X = fe_v2(lab, pd.concat([lab, unl])); NU = len(unl)


def fit(Xa, ya, Xv, yv, rs, w=None):
    m = lgb.LGBMClassifier(n_estimators=5000, learning_rate=0.05, random_state=rs, **LGB_BASE)
    m.fit(Xa, ya, sample_weight=w, eval_set=[(Xv, yv)], eval_metric="auc", callbacks=[lgb.early_stopping(200, verbose=False)]); return m


res = {k: [] for k in ["a_L_only", "b1_pseudo_hard", "b2_pseudo_soft", "c_true_labels"]}
for s in SEEDS:
    oof = {k: np.zeros(len(y)) for k in res}
    for k, (a, b) in enumerate(folds_for(y, s)):
        r = np.random.default_rng(s + k); a = r.permutation(a); cut = int(0.9 * len(a)); va = a[cut:]; tr = a[:cut]
        U, L = tr[:NU], tr[NU:]
        mL = fit(X.iloc[L], y[L], X.iloc[va], y[va], s + k); oof["a_L_only"][b] = mL.predict_proba(X.iloc[b])[:, 1]
        pu = mL.predict_proba(X.iloc[U])[:, 1]
        keep = np.abs(pu - 0.5) > 0.35
        m = fit(pd.concat([X.iloc[L], X.iloc[U[keep]]]), np.concatenate([y[L], (pu[keep] > 0.5).astype(int)]), X.iloc[va], y[va], s + k,
                np.concatenate([np.ones(len(L)), np.full(keep.sum(), 0.5)])); oof["b1_pseudo_hard"][b] = m.predict_proba(X.iloc[b])[:, 1]
        m = fit(pd.concat([X.iloc[L], X.iloc[U], X.iloc[U]]), np.concatenate([y[L], np.ones(len(U), int), np.zeros(len(U), int)]), X.iloc[va], y[va], s + k,
                np.concatenate([np.ones(len(L)), pu, 1 - pu])); oof["b2_pseudo_soft"][b] = m.predict_proba(X.iloc[b])[:, 1]
        m = fit(X.iloc[tr], y[tr], X.iloc[va], y[va], s + k); oof["c_true_labels"][b] = m.predict_proba(X.iloc[b])[:, 1]
    for kk in res:
        res[kk].append((nested_thr_score(y, oof[kk], folds_for(y, s)), roc_auc_score(y, oof[kk])))
    print("seed", s, {kk: round(v[-1][0], 5) for kk, v in res.items()}, flush=True)
base = np.array([v[0] for v in res["a_L_only"]])
for kk, v in res.items():
    f = np.array([t[0] for t in v]); au = np.array([t[1] for t in v])
    r = dict(name=f"exp19 {kk}", f2m_nested=round(f.mean(), 5), f2m_nested_sd=round(f.std(ddof=1), 5), auc=round(au.mean(), 5),
             paired_gain_vs_L_only=round((f - base).mean(), 5))
    print(json.dumps(r)); open(f"{OUT}/results_ms.jsonl", "a").write(json.dumps(r) + "\n")

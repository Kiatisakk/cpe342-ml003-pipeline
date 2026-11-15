"""In-sample ("train") vs out-of-fold scores of the final recipe, to show the overfitting gap.
Model = final_v2 recipe (fe_v2 + 4 MCAR-masked copies + LGB lr 0.02), trained on 90% of ALL labeled rows
(10% for early stopping), then scored on the same 90% rows it was trained on (in-sample) and on the 10% holdout."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import *
import lightgbm as lgb
lab, unl, te, ids = load(); y = lab[TARGET].astype(int).values
ref = pd.concat([lab, unl]); X = fe_v2(lab, ref); miss = lab[SIGNAL].isna().mean().values
rng = np.random.default_rng(0); idx = rng.permutation(len(y)); cut = int(0.9 * len(y)); tr, va = idx[:cut], idx[cut:]


def masked(df):
    d = df.copy(); M = rng.random((len(d), len(SIGNAL))) < miss[None, :]; v = d[SIGNAL].values.copy(); v[M] = np.nan; d[SIGNAL] = v
    return fe_v2(d, ref)


rows = []
for nm, nc in [("final recipe (with 4 masked copies)", 4), ("same, no augmentation", 0)]:
    Xa = pd.concat([X.iloc[tr]] + [masked(lab.iloc[tr]) for _ in range(nc)], ignore_index=True)
    m = lgb.LGBMClassifier(n_estimators=20000, learning_rate=0.02, random_state=0, **LGB_BASE)
    m.fit(Xa, np.tile(y[tr], nc + 1), eval_set=[(X.iloc[va], y[va])], eval_metric="auc", callbacks=[lgb.early_stopping(200, verbose=False)])
    for part, ii in [("train rows (in-sample)", tr), ("10% holdout", va)]:
        p = m.predict_proba(X.iloc[ii])[:, 1]
        r = dict(model=nm, eval_on=part, iters=m.best_iteration_, auc=round(roc_auc_score(y[ii], p), 4), f2m=round(best_thr(y[ii], p)[1], 4),
                 f2b=round(best_thr(y[ii], p, "binary")[1], 4))
        print(json.dumps(r), flush=True); rows.append(r)
json.dump(rows, open(f"{OUT}/exp21_insample.json", "w"), indent=1)

"""LightGBM variants with a different inductive bias on fe_v2 (lr 0.05, 3 fold-seeds); OOFs saved for blending."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import *
lab, unl, te, ids = load(); y = lab[TARGET].astype(int).values
X = fe_v2(lab, pd.concat([lab, unl]))
V = {
    "extra_trees": dict(extra_trees=True),
    "linear_tree": dict(linear_tree=True, linear_lambda=1.0),
    "goss": dict(data_sample_strategy="goss", subsample=1.0, subsample_freq=0),
    "deep_reg": dict(num_leaves=255, min_child_samples=100, reg_lambda=30, colsample_bytree=0.3),
    "shallow": dict(num_leaves=7, min_child_samples=20),
}
for k in (sys.argv[1:] or list(V)):
    a, o = multi_seed(f"exp13 lgb fe_v2 {k}", y, lambda s: cv_lgb(X, y, seed=s, params=V[k])[0]); np.save(f"{OUT}/exp13_{k}.npy", np.array(o))

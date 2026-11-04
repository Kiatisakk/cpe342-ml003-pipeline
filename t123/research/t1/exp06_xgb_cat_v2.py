"""XGB and CatBoost (baseline-tuned params) on fe_v2, 3 fold-seeds; OOFs saved for blending."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import *
lab, unl, te, ids = load(); y = lab[TARGET].astype(int).values
X = fe_v2(lab, pd.concat([lab, unl]))
for nm, fp in [("xgb", xgb_fp()), ("cat", cat_fp())]:
    if len(sys.argv) > 1 and nm not in sys.argv[1:]: continue
    a, o = multi_seed(f"exp06 {nm} fe_v2", y, lambda s: cv_generic(fp, X, y, seed=s)[0]); np.save(f"{OUT}/exp06_{nm}.npy", np.array(o))

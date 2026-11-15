"""Baseline FE + exact zone posteriors for missing age / level (single LGB, 3 fold-seeds)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import *
lab, unl, te, ids = load(); y = lab[TARGET].astype(int).values
ref = pd.concat([lab, unl])
X = fe_v2(lab, ref); print(X.shape, X.filter(like="pa_").describe().T.round(3).to_string())
agg, oofs = multi_seed("exp03 lgb fe_v2 (zone posteriors)", y, lambda s: cv_lgb(X, y, seed=s)[0])
np.save(f"{OUT}/exp03.npy", np.array(oofs))

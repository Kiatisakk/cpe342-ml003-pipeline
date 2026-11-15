"""Multi-seed (fold split 42/7/2024) comparison: baseline FE vs + noise features (single LGB)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import *
lab, unl, te, ids = load(); y = lab[TARGET].astype(int).values
ref = pd.concat([lab, unl]); noise = [c for c in raw_cols(lab) if c not in SIGNAL]
A = fe_baseline(lab, ref); B = pd.concat([A, lab[noise]], axis=1)
for nm, X in [("fe_base", A), ("fe_base+noise16", B)][:: (1 if len(sys.argv) < 2 else -1)]:
    agg, oofs = multi_seed(f"exp02 lgb {nm}", y, lambda s: cv_lgb(X, y, seed=s)[0])
    np.save(f"{OUT}/exp02_{nm}.npy", np.array(oofs))

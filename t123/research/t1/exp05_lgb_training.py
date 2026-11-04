"""Training tweaks on fe_v2 with LGB: lower LR, class weighting, in-fold multi-seed bagging."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import *
lab, unl, te, ids = load(); y = lab[TARGET].astype(int).values
X = fe_v2(lab, pd.concat([lab, unl]))
which = sys.argv[1:] or ["lr02", "spw2", "bag5"]
if "lr02" in which:
    a, o = multi_seed("exp05 lgb fe_v2 lr0.02", y, lambda s: cv_lgb(X, y, seed=s, lr=0.02)[0]); np.save(f"{OUT}/exp05_lr02.npy", np.array(o))
if "spw2" in which:
    a, o = multi_seed("exp05 lgb fe_v2 scale_pos_weight=1.86 (balanced)", y, lambda s: cv_lgb(X, y, seed=s, params=dict(scale_pos_weight=1.86))[0])
if "bag5" in which:
    def bag(s):
        return np.mean([cv_lgb(X, y, seed=s, inner_seed_offset=100 * i, params=dict(seed=1000 * i + s))[0] for i in range(5)], 0)
    a, o = multi_seed("exp05 lgb fe_v2 bag5 (inner split + model seed)", y, bag); np.save(f"{OUT}/exp05_bag5.npy", np.array(o))

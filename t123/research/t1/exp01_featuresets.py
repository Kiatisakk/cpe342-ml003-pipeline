"""Feature-set screening with one LightGBM (baseline-tuned params, lr 0.05, 5-fold seed 42)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import *
lab, unl, te, ids = load(); y = lab[TARGET].astype(int).values
ref = pd.concat([lab, unl])
noise = [c for c in raw_cols(lab) if c not in SIGNAL]
sets = {
    "raw15": lab[SIGNAL],
    "raw31": lab[raw_cols(lab)],
    "fe_base": fe_baseline(lab, ref),
    "fe_base+noise16": pd.concat([fe_baseline(lab, ref), lab[noise]], axis=1),
}
want = sys.argv[1:] or list(sets)
for k in want:
    T = Timer(); oof, _, its = cv_lgb(sets[k], y, ret_iters=True)
    np.save(f"{OUT}/exp01_{k}.npy", oof)
    report(f"exp01 lgb {k}", y, oof, extra=dict(nfeat=sets[k].shape[1], iters=its, time=T()))

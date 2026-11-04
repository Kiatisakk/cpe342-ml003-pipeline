"""fe_v3 = fe_v2 + label-free multiclass imputers (zone probabilities + expected value) for missing
reports / friends / crosshair / age / level, trained on TRAIN rows (labeled + unlabeled) with all 30 other raw columns."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import *
lab, unl, te, ids = load(); y = lab[TARGET].astype(int).values
ref = pd.concat([lab, unl]).reset_index(drop=True)
use_noise = "nonoise" not in sys.argv
T = Timer(); imps = fit_imputers(ref, use_noise=use_noise); print("imputers", T(), flush=True)
X = fe_v3(lab, ref, imps); print(X.shape, flush=True)
for f in IMPUTE_SPECS:   # how informative is the imputation itself (rows where f is missing)
    m = lab[f].isna().values
    print(f, "label AUC of imputed value on missing rows", round(roc_auc_score(y[m], X.loc[m, f"imp_{f}"]), 4))
tag = "" if use_noise else " (imputers w/o noise cols)"
a, o = multi_seed("exp12 lgb fe_v3 imputers" + tag, y, lambda s: cv_lgb(X, y, seed=s)[0])
np.save(f"{OUT}/exp12{'' if use_noise else '_nonoise'}.npy", np.array(o))

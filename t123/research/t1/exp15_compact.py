"""Feature-subset ablation on fe_v2: does removing the baseline's weaker engineered groups help (less noise)?"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import *
lab, unl, te, ids = load(); y = lab[TARGET].astype(int).values
X = fe_v2(lab, pd.concat([lab, unl]))
groups = {
    "no_freq_logs": [c for c in X.columns if c.startswith(("freq_", "log_"))],
    "no_skillz_inter": ["skill_z_sum", "skill_z_std", "kd_x_hs", "acc_x_cross", "reports_per_100d", "friends_per_100d", "level_per_100d"],
    "compact": [c for c in X.columns if c.startswith(("freq_", "log_", "na_")) or c in ("skill_z_sum", "skill_z_std", "kd_x_hs", "acc_x_cross",
                "reports_per_100d", "friends_per_100d", "level_per_100d", "age_segment", "n_missing", "zone_cheat", "zone_legit")],
}
for k in (sys.argv[1:] or list(groups)):
    Xk = X.drop(columns=groups[k])
    a, o = multi_seed(f"exp15 lgb fe_v2 {k} ({Xk.shape[1]} feats)", y, lambda s: cv_lgb(Xk, y, seed=s)[0])

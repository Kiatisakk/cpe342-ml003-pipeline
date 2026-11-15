"""fe_v2 + ratio / interaction features among signal features (course hint: game_sense / accuracy)."""
import sys, os, itertools
sys.path.insert(0, os.path.dirname(__file__))
from common import *
lab, unl, te, ids = load(); y = lab[TARGET].astype(int).values
ref = pd.concat([lab, unl]); X = fe_v2(lab, ref)
skill = ["kill_death_ratio", "headshot_percentage", "win_rate", "accuracy_score", "kill_consistency", "reaction_time_ms",
         "aiming_smoothness", "spray_control_score", "game_sense_score", "crosshair_placement"]
R = pd.DataFrame(index=lab.index)
R["skill_consistency"] = lab["game_sense_score"] / lab["accuracy_score"]
Xh = pd.concat([X, R], axis=1)
for a, b in itertools.combinations(skill, 2):
    R[f"r_{a[:6]}_{b[:6]}"] = np.log(lab[a] / lab[b])
R["fr_x_age"] = lab["friend_network_size"] / (X["age_post_mean"] + 1)
R["rep_x_fr"] = lab["reports_received"] / (lab["friend_network_size"] + 1)
Xall = pd.concat([X, R], axis=1)
which = sys.argv[1:] or ["hint", "all"]
if "hint" in which:
    a, o = multi_seed("exp08 lgb fe_v2 + game_sense/accuracy", y, lambda s: cv_lgb(Xh, y, seed=s)[0])
if "all" in which:
    a, o = multi_seed(f"exp08 lgb fe_v2 + 45 log-ratios + 3 ({Xall.shape[1]} feats)", y, lambda s: cv_lgb(Xall, y, seed=s)[0])

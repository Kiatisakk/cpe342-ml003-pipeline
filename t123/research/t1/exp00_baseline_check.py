"""Re-score the baseline OOF blend with the research harness (incl. nested threshold)."""
import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import *
lab, unl, te, ids = load(); y = lab[TARGET].astype(int).values
base = np.load(os.path.join(ROOT, "..", "..", "preds_final", "t1", "t1_pipeline", "oof_blend.npy"))
t, s, r = best_thr(y, base)
print("sklearn check", f2m(y, (base >= t).astype(int)), s)
report("baseline_blend(xgb+lgb+cat, repo)", y, base)
for m in ["xgb", "lgb", "cat"]:
    report(f"baseline_{m}(repo)", y, np.load(os.path.join(ROOT, "..", "..", "preds_final", "t1", "t1_pipeline", f"oof_{m}.npy")))

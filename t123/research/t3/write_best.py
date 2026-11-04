# Write best_oof.csv / t3_best_test.csv from a saved OOF name (oof/NAME_oof.npy, NAME_test.npy).
# usage: python t123/research/t3/write_best.py NAME
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load, nmae, fold_scores, load_oof, HERE

name = sys.argv[1]
D = load(); y = D["y"]
o, t = load_oof(name); o, t = np.clip(o, 0, None), np.clip(t, 0, None)
assert len(t) == 25889 and np.isfinite(t).all()
pd.DataFrame({"y": y, "pred": o}).to_csv(os.path.join(HERE, "best_oof.csv"), index=False)
pd.DataFrame({"id": D["ids"], "pred": t}).to_csv(os.path.join(HERE, "t3_best_test.csv"), index=False)
m, s = fold_scores(y, o, D["folds"])
print(f"wrote best from {name}: OOF {nmae(y, o):.5f} fold {m:.5f} +- {s:.5f} | test mean {t.mean():.1f} zero% {(t == 0).mean():.3f}")

# Mix general and whale-specialist positive-part quantiles by p_whale, hurdle median. Nested-free (no tuning).
# usage: python t3/research/mix_whale.py GENERAL_Q WHALE_Q OUT [--p lgbh_log] [--mode pw|hard]
import sys, os, argparse, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load, nmae, fold_scores, save_oof, hurdle_median
ap = argparse.ArgumentParser(); ap.add_argument("g"); ap.add_argument("w"); ap.add_argument("out")
ap.add_argument("--p", default="lgbh_log"); ap.add_argument("--lam", type=float, default=1.0)
a = ap.parse_args(); OD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "oof")
ALPHAS = np.array([0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
D = load(); y, F = D["y"], D["folds"]; n = len(y)
G, W, P = (np.load(os.path.join(OD, f)) for f in (f"{a.g}_Q.npy", f"{a.w}_Q.npy", f"{a.p}_p.npy"))
pw = a.lam * np.r_[D["X"]["p_whale"].values, D["Xt"]["p_whale"].values][:, None]
M = np.exp((1 - pw) * np.log(G) + pw * np.log(W))       # geometric mix of quantiles
o = hurdle_median(P[:n], M[:n], ALPHAS); t = hurdle_median(P[n:], M[n:], ALPHAS)
save_oof(a.out, o, t); np.save(os.path.join(OD, f"{a.out}_Q.npy"), M)
m, s = fold_scores(y, o, F); seg = D["X"]["segment"].values
print(f"RESULT {a.out}: OOF {nmae(y, o):.5f} fold {m:.5f} +- {s:.5f}")

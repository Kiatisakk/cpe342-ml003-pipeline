# Average positive-part quantiles (geometric) across saved models and/or classifier p across seeds, then hurdle median.
# usage: python t123/research/t3/avg_q.py OUT --q name1,name2,... --p pname1,pname2,...
import sys, os, argparse, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load, nmae, fold_scores, save_oof, hurdle_median
ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--q"); ap.add_argument("--p", default="lgbh_log")
a = ap.parse_args(); OD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "oof")
ALPHAS = np.array([0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
D = load(); y, F = D["y"], D["folds"]; n = len(y)
Q = np.exp(np.mean([np.log(np.load(os.path.join(OD, f"{q}_Q.npy"))) for q in a.q.split(",")], 0))
P = np.mean([np.load(os.path.join(OD, f"{p}_p.npy")) for p in a.p.split(",")], 0)
o = hurdle_median(P[:n], Q[:n], ALPHAS); t = hurdle_median(P[n:], Q[n:], ALPHAS)
save_oof(a.out, o, t); np.save(os.path.join(OD, f"{a.out}_Q.npy"), Q); np.save(os.path.join(OD, f"{a.out}_p.npy"), P)
m, s = fold_scores(y, o, F)
print(f"RESULT {a.out}: OOF {nmae(y, o):.5f} fold {m:.5f} +- {s:.5f}")

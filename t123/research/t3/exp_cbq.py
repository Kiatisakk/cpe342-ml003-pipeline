# CatBoost MultiQuantile on positive part, target log y - log prev (offset), hurdle median with saved p.
# usage: python t123/research/t3/exp_cbq.py NAME [--depth 8] [--lr 0.06] [--seed 0] [--p lgbh_log]
import sys, os, time, argparse, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load, nmae, fold_scores, save_oof, hurdle_median, extra_feats, prev_offset, NT
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
ap = argparse.ArgumentParser(); ap.add_argument("name"); ap.add_argument("--depth", type=int, default=8)
ap.add_argument("--lr", type=float, default=0.06); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--p", default="lgbh_log")
a = ap.parse_args()
ALPHAS = np.array([0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
OD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "oof")
D = load(); X, Xt = extra_feats(D["X"], D["Xt"]); y, F = D["y"], D["folds"]; n = len(y)
pp = np.load(os.path.join(OD, f"{a.p}_p.npy")); Op, Tp = pp[:n], pp[n:]
OQ = np.zeros((n, len(ALPHAS))); TQ = np.zeros((len(Xt), len(ALPHAS))); t0 = time.time(); its = []
loss = "MultiQuantile:alpha=" + ",".join(str(x) for x in ALPHAS)
for k, (tr, va) in enumerate(F):
    Xa, ya = X.iloc[tr], y[tr]; pos = ya > 0; Xp = Xa[pos]; r = np.log(ya[pos]) - prev_offset(Xp)
    i, j = train_test_split(np.arange(len(r)), test_size=0.1, random_state=a.seed)
    m = CatBoostRegressor(loss_function=loss, iterations=20000, learning_rate=a.lr, depth=a.depth, l2_leaf_reg=3,
                          od_type="Iter", od_wait=200, random_seed=a.seed, thread_count=NT, verbose=False)
    m.fit(Xp.iloc[i], r[i], eval_set=(Xp.iloc[j], r[j]), use_best_model=True); its.append(m.get_best_iteration())
    OQ[va] = np.exp(np.sort(m.predict(X.iloc[va]), 1) + prev_offset(X.iloc[va])[:, None])
    TQ += np.exp(np.sort(m.predict(Xt), 1) + prev_offset(Xt)[:, None]) / 5
    print(f"fold {k} {time.time()-t0:.0f}s it {its[-1]}", flush=True)
o = hurdle_median(Op, OQ, ALPHAS); t = hurdle_median(Tp, TQ, ALPHAS)
save_oof(a.name, o, t); np.save(os.path.join(OD, f"{a.name}_Q.npy"), np.r_[OQ, TQ])
mm, s = fold_scores(y, o, F)
print(f"RESULT {a.name}: OOF {nmae(y, o):.5f} fold mean {mm:.5f} sd {s:.5f} | iters {its}")

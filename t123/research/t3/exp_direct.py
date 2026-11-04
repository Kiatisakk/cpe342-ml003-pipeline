# Direct MAE model on RAW y (baht scale) over ALL rows (zeros included), log link with offset:
#   f(x) = exp(log_prev(x) + g(x)),  loss = |y - f|,  grad = -sign(y - f) * f,  hess = f  (scaled sign-descent in log space)
# The baht-weighted gradient matches the NMAE metric directly; the conditional median of y (mixture incl. zeros) is the optimum.
# usage: python t123/research/t3/exp_direct.py NAME [--lr 0.05] [--depth 7] [--seed 0] [--floor 0]
import sys, os, time, argparse, numpy as np, xgboost as xgb
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load, nmae, fold_scores, save_oof, extra_feats, prev_offset, NT
from sklearn.model_selection import train_test_split
ap = argparse.ArgumentParser(); ap.add_argument("name"); ap.add_argument("--lr", type=float, default=0.05)
ap.add_argument("--depth", type=int, default=7); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--mcw", type=float, default=5); ap.add_argument("--p", default=None, help="optional p feature from saved clf")
a = ap.parse_args()
D = load(); X, Xt = extra_feats(D["X"], D["Xt"]); y, F = D["y"], D["folds"]; n = len(y)
if a.p:
    pp = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "oof", f"{a.p}_p.npy"))
    X["p_clf"], Xt["p_clf"] = pp[:n], pp[n:]    # OOF p for train rows, fold-avg p for test (stacking feature)

def obj(pred, dtrain):
    yy = dtrain.get_label(); f = np.exp(np.clip(pred, -20, 14))
    return -np.sign(yy - f) * f, np.maximum(f, 1e-3)

def mae_eval(pred, dtrain):
    yy = dtrain.get_label(); return "nmae_err", float(np.abs(yy - np.exp(np.clip(pred, -20, 14))).sum() / yy.sum())

O = np.zeros(n); T = np.zeros(len(Xt)); t0 = time.time(); its = []
off_all, off_te = prev_offset(X), prev_offset(Xt)
dte = xgb.DMatrix(Xt, base_margin=off_te)
params = dict(eta=a.lr, max_depth=a.depth, min_child_weight=a.mcw, subsample=0.8, colsample_bytree=0.7, reg_lambda=2.0,
              tree_method="hist", nthread=NT, seed=a.seed, disable_default_eval_metric=1)
for k, (tr, va) in enumerate(F):
    i, j = train_test_split(tr, test_size=0.1, random_state=a.seed)
    dtr = xgb.DMatrix(X.iloc[i], label=y[i], base_margin=off_all[i]); des = xgb.DMatrix(X.iloc[j], label=y[j], base_margin=off_all[j])
    b = xgb.train(params, dtr, 20000, evals=[(des, "es")], obj=obj, custom_metric=mae_eval, early_stopping_rounds=200, verbose_eval=False)
    its.append(b.best_iteration); rng = (0, b.best_iteration + 1)
    dva = xgb.DMatrix(X.iloc[va], base_margin=off_all[va])
    O[va] = np.exp(b.predict(dva, output_margin=True, iteration_range=rng)); T += np.exp(b.predict(dte, output_margin=True, iteration_range=rng)) / 5
    print(f"fold {k} {time.time()-t0:.0f}s it {its[-1]} fold score {nmae(y[va], O[va]):.5f}", flush=True)
O, T = np.clip(O, 0, 300000), np.clip(T, 0, 300000)
save_oof(a.name, O, T)
m, s = fold_scores(y, O, F)
print(f"RESULT {a.name}: OOF {nmae(y, O):.5f} fold mean {m:.5f} sd {s:.5f} | iters {its} | frac<1 {np.mean(O < 1):.3f}")

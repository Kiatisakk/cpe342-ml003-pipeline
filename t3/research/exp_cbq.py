# CatBoost MultiQuantile on positive part, target log y - log prev (offset), hurdle median with saved p.
# usage: python t3/research/exp_cbq.py NAME [--depth 8] [--lr 0.06] [--seed 0] [--p lgbh_log]
import sys, os, time, argparse, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load, nmae, fold_scores, save_oof, hurdle_median, extra_feats, prev_offset, NT
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
ap = argparse.ArgumentParser(); ap.add_argument("name"); ap.add_argument("--depth", type=int, default=8)
ap.add_argument("--lr", type=float, default=0.06); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--p", default="lgbh_log")
ap.add_argument("--alphas", default="0.02,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5")
ap.add_argument("--l2", type=float, default=3); ap.add_argument("--border", type=int, default=254)
ap.add_argument("--rs", type=float, default=1.0, help="random_strength")
ap.add_argument("--ratio", action="store_true", help="add features in ratio space relative to prev_month")
ap.add_argument("--frac", type=float, default=1.0, help="learning curve: fraction of training-fold payers used")
ap.add_argument("--bt", type=float, default=1.0, help="bagging_temperature")
ap.add_argument("--rsm", type=float, default=1.0); ap.add_argument("--grow", default="SymmetricTree"); ap.add_argument("--mdl", type=int, default=1)
ap.add_argument("--p_avg", action="store_true", help="score with 3-seed p (lgbh_log, clf_s1, clf_s2)")
a = ap.parse_args()
ALPHAS = np.array([float(x) for x in a.alphas.split(",")])
OD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "oof")
D = load(); X, Xt = extra_feats(D["X"], D["Xt"]); y, F = D["y"], D["folds"]; n = len(y)
if a.ratio:
    for Z in (X, Xt):
        o = prev_offset(Z)
        for c in ["log_historical_spending", "log_avg_transaction_value", "l_atv_x_freq", "l_hist_per_tx", "l_hist_per_age"]:
            Z[f"r_{c}"] = Z[c] - o
        Z["r_tx"] = np.log1p(Z["total_transactions"]) - o
        Z["r_disc"] = np.log1p(Z["purchases_on_discount"]) - o
pnames = ["lgbh_log", "clf_s1", "clf_s2"] if a.p_avg else [a.p]
pp = np.mean([np.load(os.path.join(OD, f"{q}_p.npy")) for q in pnames], 0); Op, Tp = pp[:n], pp[n:]
OQ = np.zeros((n, len(ALPHAS))); TQ = np.zeros((len(Xt), len(ALPHAS))); t0 = time.time(); its = []
loss = "MultiQuantile:alpha=" + ",".join(str(x) for x in ALPHAS)
for k, (tr, va) in enumerate(F):
    Xa, ya = X.iloc[tr], y[tr]; pos = ya > 0; Xp = Xa[pos]; r = np.log(ya[pos]) - prev_offset(Xp)
    if a.frac < 1:
        keep = np.random.RandomState(a.seed + k).rand(len(r)) < a.frac; Xp, r = Xp[keep], r[keep]
    i, j = train_test_split(np.arange(len(r)), test_size=0.1, random_state=a.seed)
    m = CatBoostRegressor(loss_function=loss, iterations=20000, learning_rate=a.lr, depth=a.depth, l2_leaf_reg=a.l2, border_count=a.border, random_strength=a.rs, bagging_temperature=a.bt, rsm=a.rsm,
                          grow_policy=a.grow, **({"min_data_in_leaf": a.mdl} if a.grow != "SymmetricTree" else {}),
                          od_type="Iter", od_wait=200, random_seed=a.seed, thread_count=NT, verbose=False)
    m.fit(Xp.iloc[i], r[i], eval_set=(Xp.iloc[j], r[j]), use_best_model=True); its.append(m.get_best_iteration())
    OQ[va] = np.exp(np.sort(m.predict(X.iloc[va]), 1) + prev_offset(X.iloc[va])[:, None])
    TQ += np.exp(np.sort(m.predict(Xt), 1) + prev_offset(Xt)[:, None]) / 5
    print(f"fold {k} {time.time()-t0:.0f}s it {its[-1]}", flush=True)
o = hurdle_median(Op, OQ, ALPHAS); t = hurdle_median(Tp, TQ, ALPHAS)
save_oof(a.name, o, t); np.save(os.path.join(OD, f"{a.name}_Q.npy"), np.r_[OQ, TQ]); np.save(os.path.join(OD, f"{a.name}_alphas.npy"), ALPHAS)
mm, s = fold_scores(y, o, F)
print(f"RESULT {a.name}: OOF {nmae(y, o):.5f} fold mean {mm:.5f} sd {s:.5f} | iters {its}")

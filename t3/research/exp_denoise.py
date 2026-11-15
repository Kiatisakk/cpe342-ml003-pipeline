# Noise removal from TRAINING data only (validation folds keep all rows, like the real test set).
# Nested: inside each outer training fold, a 3-fold inner CV of the CatBoost d4 offset-quantile model gives an inner-OOF median
# for every training payer; rows with the largest |log residual| are treated as noise. Then the amount model is refit on the
# cleaned training fold and scored on the untouched validation fold (hurdle median with the fixed 3-seed classifier p).
# Variants: drop top 0% (control, same code path) / 1% / 3% / 5% / 10%, and Cauchy down-weighting w = 1/(1+(res/c)^2).
# usage: python t3/research/exp_denoise.py
import sys, os, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load, nmae, fold_scores, save_oof, hurdle_median, extra_feats, prev_offset, NT, HERE
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split, KFold
ALPHAS = np.array([0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]); MED = len(ALPHAS) - 1
OD = os.path.join(HERE, "oof")
D = load(); X, Xt = extra_feats(D["X"], D["Xt"]); y, F = D["y"], D["folds"]; n = len(y)
P = np.mean([np.load(os.path.join(OD, f"{q}_p.npy")) for q in ("lgbh_log", "clf_s1", "clf_s2")], 0)
loss = "MultiQuantile:alpha=" + ",".join(str(x) for x in ALPHAS)


def fit(Xa, r, w=None, seed=0):
    i, j = train_test_split(np.arange(len(r)), test_size=0.1, random_state=seed)
    m = CatBoostRegressor(loss_function=loss, iterations=20000, learning_rate=0.05, depth=4, l2_leaf_reg=3, od_type="Iter", od_wait=200,
                          random_seed=seed, thread_count=NT, verbose=False)
    m.fit(Xa.iloc[i], r[i], sample_weight=None if w is None else w[i], eval_set=(Xa.iloc[j], r[j]), use_best_model=True)
    return m


VARIANTS = ["drop0", "drop1", "drop3", "drop5", "drop10", "cauchy"]
OQ = {v: np.zeros((n, len(ALPHAS))) for v in VARIANTS}; TQ = {v: np.zeros((len(Xt), len(ALPHAS))) for v in VARIANTS}
t0 = time.time(); stats = []
for k, (tr, va) in enumerate(F):
    pos = tr[y[tr] > 0]; Xp = X.iloc[pos]; r = np.log(y[pos]) - prev_offset(Xp)
    res = np.zeros(len(r))                                  # inner-OOF residual vs inner median, training fold only
    for itr, iva in KFold(3, shuffle=True, random_state=k).split(r):
        m = fit(Xp.iloc[itr], r[itr]); res[iva] = r[iva] - m.predict(Xp.iloc[iva])[:, MED]
    a = np.abs(res); c = 1.4826 * np.median(a)               # robust scale (MAD)
    stats.append((round(float(c), 3), round(float(np.quantile(a, 0.99)), 3), int((a > 4 * c).sum())))
    for v in VARIANTS:
        w = None; keep = np.ones(len(r), bool)
        if v.startswith("drop") and v != "drop0": keep = a <= np.quantile(a, 1 - int(v[4:]) / 100)
        if v == "cauchy": w = 1 / (1 + (res / (2 * c)) ** 2)
        m = fit(Xp[keep], r[keep], None if w is None else w)
        OQ[v][va] = np.exp(np.sort(m.predict(X.iloc[va]), 1) + prev_offset(X.iloc[va])[:, None])
        TQ[v] += np.exp(np.sort(m.predict(Xt), 1) + prev_offset(Xt)[:, None]) / 5
    print(f"fold {k} {time.time()-t0:.0f}s | inner-res MAD-sd {stats[-1][0]} q99 {stats[-1][1]} n>4sd {stats[-1][2]}", flush=True)
ref = None
for v in VARIANTS:
    o = hurdle_median(P[:n], OQ[v], ALPHAS); t = hurdle_median(P[n:], TQ[v], ALPHAS)
    save_oof(f"dn_{v}", o, t); np.save(os.path.join(OD, f"dn_{v}_Q.npy"), np.r_[OQ[v], TQ[v]])
    if ref is None: ref = o
    d = np.array([nmae(y[va], o[va]) - nmae(y[va], ref[va]) for _, va in F]); m_, s_ = fold_scores(y, o, F)
    print(f"RESULT dn_{v}: OOF {nmae(y, o):.5f} fold {m_:.5f} +- {s_:.5f} | paired vs drop0 {d.mean():+.5f} +- {d.std():.5f}", flush=True)

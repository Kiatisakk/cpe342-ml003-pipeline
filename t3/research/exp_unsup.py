# Unsupervised views of the data (no target used to decide), all fitted on the TRAINING fold only (never on test, never on the
# validation fold); validation folds keep every row. Amount model = CatBoost d4 offset multi-quantile, fixed 3-seed classifier p.
#   ctrl      : same code path, nothing removed
#   iso1/3/5  : IsolationForest on standardized, median-imputed features -> drop the 1/3/5 % most anomalous training payers
#   maha3     : Mahalanobis distance in standardized PCA space (95 % var) -> drop the 3 % farthest training payers
#   dropfeat  : drop features whose CatBoost importance in the ctrl model is < 0.5 % (candidate noise dimensions), refit
#   pca       : add the first 8 PCA components (fitted on the training fold) as extra features
# usage: python t3/research/exp_unsup.py
import sys, os, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load, nmae, fold_scores, save_oof, hurdle_median, extra_feats, prev_offset, NT, HERE
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
ALPHAS = np.array([0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
OD = os.path.join(HERE, "oof")
D = load(); X, Xt = extra_feats(D["X"], D["Xt"]); y, F = D["y"], D["folds"]; n = len(y)
P = np.mean([np.load(os.path.join(OD, f"{q}_p.npy")) for q in ("lgbh_log", "clf_s1", "clf_s2")], 0)
loss = "MultiQuantile:alpha=" + ",".join(str(x) for x in ALPHAS)


def fit(Xa, r):
    i, j = train_test_split(np.arange(len(r)), test_size=0.1, random_state=0)
    m = CatBoostRegressor(loss_function=loss, iterations=20000, learning_rate=0.05, depth=4, l2_leaf_reg=3, od_type="Iter", od_wait=200,
                          random_seed=0, thread_count=NT, verbose=False)
    m.fit(Xa.iloc[i], r[i], eval_set=(Xa.iloc[j], r[j]), use_best_model=True); return m


VARIANTS = ["ctrl", "iso1", "iso3", "iso5", "maha3", "dropfeat", "pca"]
OQ = {v: np.zeros((n, len(ALPHAS))) for v in VARIANTS}; TQ = {v: np.zeros((len(Xt), len(ALPHAS))) for v in VARIANTS}
t0 = time.time(); dropped_log = []
for k, (tr, va) in enumerate(F):
    pos = tr[y[tr] > 0]; Xp = X.iloc[pos]; r = np.log(y[pos]) - prev_offset(Xp)
    med, mu, sd = Xp.median(), None, None
    Z = Xp.fillna(med); mu, sd = Z.mean(), Z.std() + 1e-9
    std = lambda A: ((A.fillna(med) - mu) / sd).values
    Zs = std(Xp)
    iso = IsolationForest(n_estimators=300, random_state=k, n_jobs=NT).fit(Zs); s_iso = -iso.score_samples(Zs)   # high = anomalous
    pca = PCA(n_components=0.95, random_state=0).fit(Zs); C = pca.transform(Zs); s_mah = ((C / np.sqrt(pca.explained_variance_)) ** 2).sum(1)
    pca8 = PCA(n_components=8, random_state=0).fit(Zs)
    def add_pca(A):
        A = A.copy(); Cm = pca8.transform(std(A))
        for c in range(8): A[f"pc{c}"] = Cm[:, c]
        return A
    for v in VARIANTS:
        keep = np.ones(len(r), bool); Xa, Xv, Xte = Xp, X.iloc[va], Xt
        if v.startswith("iso"): keep = s_iso <= np.quantile(s_iso, 1 - int(v[3:]) / 100)
        if v == "maha3": keep = s_mah <= np.quantile(s_mah, 0.97)
        if v == "dropfeat":
            imp = ctrl_model.get_feature_importance(); cols = [c for c, w in zip(Xp.columns, imp) if w >= 0.5]
            dropped_log.append(sorted(set(Xp.columns) - set(cols))); Xa, Xv, Xte = Xp[cols], X.iloc[va][cols], Xt[cols]
        if v == "pca": Xa, Xv, Xte = add_pca(Xp), add_pca(X.iloc[va]), add_pca(Xt)
        m = fit(Xa[keep], r[keep])
        if v == "ctrl": ctrl_model = m
        OQ[v][va] = np.exp(np.sort(m.predict(Xv), 1) + prev_offset(X.iloc[va])[:, None])
        TQ[v] += np.exp(np.sort(m.predict(Xte), 1) + prev_offset(Xt)[:, None]) / 5
    print(f"fold {k} {time.time()-t0:.0f}s | dropped features: {dropped_log[-1]}", flush=True)
ref = None
for v in VARIANTS:
    o = hurdle_median(P[:n], OQ[v], ALPHAS); t = hurdle_median(P[n:], TQ[v], ALPHAS)
    save_oof(f"us_{v}", o, t); np.save(os.path.join(OD, f"us_{v}_Q.npy"), np.r_[OQ[v], TQ[v]])
    if ref is None: ref = o
    d = np.array([nmae(y[va], o[va]) - nmae(y[va], ref[va]) for _, va in F]); m_, s_ = fold_scores(y, o, F)
    print(f"RESULT us_{v}: OOF {nmae(y, o):.5f} fold {m_:.5f} +- {s_:.5f} | paired vs ctrl {d.mean():+.5f} +- {d.std():.5f}", flush=True)

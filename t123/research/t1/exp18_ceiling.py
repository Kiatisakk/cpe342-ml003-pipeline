"""How far is 0.79 F2-macro? (1) AUC needed: blend the final OOF score with label information until the
OOF-optimal F2m hits a target, report the AUC there (overall and inside the non-zone 'core').
(2) Learning curve: LGB fe_v2 (lr 0.05) trained on a fraction of each fold-train set, scored on the full held-out folds,
3 split seeds; power-law fit s(n) = s_inf - a * n^-b extrapolates the score with unlimited data."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from scipy.optimize import curve_fit
lab, unl, te, ids = load(); y = lab[TARGET].astype(int).values
avg = np.load(os.path.join(ROOT, "best_oof_avg3.npy"))
rep, ch, fr, lv, ag = (lab[c] for c in ["reports_received", "crosshair_placement", "friend_network_size", "level", "account_age_days"])
zone = ((rep == 4) | (ch < 0.82) | (lv < 28) | ((ag >= 110) & (ag < 120)) | (rep >= 20) | (ch > 0.94) | (lv > 96) | (ag < 15)
        | ((ag >= 180) & (ag < 200)) | (ag >= 1000) | (fr < 10) | (fr > 269)).values
core = ~zone
print(f"zone rows {zone.sum()} ({zone.mean():.1%}), core rows {core.sum()}; core cheater rate {y[core].mean():.3f}")
print(f"now: AUC {roc_auc_score(y, avg):.4f} | core AUC {roc_auc_score(y[core], avg[core]):.4f} | F2m {best_thr(y, avg)[1]:.4f} | binF2 {best_thr(y, avg, 'binary')[1]:.4f}")
rng = np.random.default_rng(0); lg = np.log(avg / (1 - avg))
print("\n(1) score improvement needed (logit + eps * label signal):")
for eps in [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]:
    q = lg + eps * (2 * y - 1) * rng.exponential(1, len(y))
    print(f"  eps {eps:.2f}: AUC {roc_auc_score(y, q):.4f} core AUC {roc_auc_score(y[core], q[core]):.4f} F2m {best_thr(y, q)[1]:.4f} binF2 {best_thr(y, q, 'binary')[1]:.4f}", flush=True)

print("\n(2) learning curve:")
fr_list = [0.1, 0.2, 0.35, 0.5, 0.7, 1.0]
X = fe_v2(lab, pd.concat([lab, unl])); import lightgbm as lgb
res = {}
for f in fr_list:
    sc, au = [], []
    for s in SEEDS:
        oof = np.zeros(len(y))
        for k, (a, b) in enumerate(folds_for(y, s)):
            r = np.random.default_rng(s + k); a = r.permutation(a)[: int(len(a) * f)]; cut = int(0.9 * len(a))
            m = lgb.LGBMClassifier(n_estimators=5000, learning_rate=0.05, random_state=s + k, **LGB_BASE)
            m.fit(X.iloc[a[:cut]], y[a[:cut]], eval_set=[(X.iloc[a[cut:]], y[a[cut:]])], eval_metric="auc", callbacks=[lgb.early_stopping(200, verbose=False)])
            oof[b] = m.predict_proba(X.iloc[b])[:, 1]
        sc.append(nested_thr_score(y, oof, folds_for(y, s))); au.append(roc_auc_score(y, oof))
    n = int(0.8 * len(y) * f); res[n] = (np.mean(sc), np.mean(au))
    print(f"  frac {f:.2f} (n_train ~{n}): nested F2m {np.mean(sc):.4f} +- {np.std(sc, ddof=1):.4f} | AUC {np.mean(au):.4f}", flush=True)
ns = np.array(list(res)); F = np.array([v[0] for v in res.values()]); A = np.array([v[1] for v in res.values()])
pl = lambda n, s, a, b: s - a * n ** (-b)
for nm, v in [("F2m", F), ("AUC", A)]:
    try:
        p, _ = curve_fit(pl, ns, v, p0=[v[-1] + 0.005, 1.0, 0.5], maxfev=20000)
        print(f"  power-law {nm}: s_inf {p[0]:.4f} | at 2x data {pl(2 * ns[-1], *p):.4f} | at 10x data {pl(10 * ns[-1], *p):.4f}")
    except Exception as e:
        print("fit failed", nm, e)

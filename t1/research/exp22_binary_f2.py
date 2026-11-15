"""Training aimed at BINARY F2 (positive class). Operating point: ~63% predicted positive, recall ~0.974,
so what matters is the high-recall tail of the ranking. Proxy metric: pAUC_hr = standardised partial AUC for
TPR >= 0.9 (= roc_auc_score(1-y, -p, max_fpr=0.1)).
Variants (fe_v2, LGB lr 0.05, 5-fold x seeds 42/7/2024, nested binary-F2 threshold):
  base      AUC early stopping (= exp03)
  spw3/spw6 scale_pos_weight 3 / 6
  es_hr     early stopping on pAUC_hr instead of AUC
  spw3_es   both
  focal     focal loss (gamma 1.5, alpha 0.75), custom objective
usage: python exp22_binary_f2.py [variants...]
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import *
import lightgbm as lgb
from scipy.special import expit
lab, unl, te, ids = load(); y = lab[TARGET].astype(int).values
X = fe_v2(lab, pd.concat([lab, unl]))


def pauc_hr(yt, p): return roc_auc_score(1 - yt, -p, max_fpr=0.1)
def lgb_pauc(yt, p): return "pauc_hr", pauc_hr(yt, p), True


def focal_obj(gamma=1.5, alpha=0.75):
    def f(yt, raw):
        p = expit(raw); eps = 1e-6
        # gradient / hessian of focal loss wrt raw score (numerical-stable closed form via finite differences on p)
        def loss(z):
            q = np.clip(expit(z), eps, 1 - eps)
            return -(alpha * yt * (1 - q) ** gamma * np.log(q) + (1 - alpha) * (1 - yt) * q ** gamma * np.log(1 - q))
        h = 1e-3
        l0, lp, lm = loss(raw), loss(raw + h), loss(raw - h)
        g = (lp - lm) / (2 * h); H = np.maximum((lp - 2 * l0 + lm) / h ** 2, 1e-6)
        return g, H
    return f


V = {"base": dict(), "spw3": dict(p=dict(scale_pos_weight=3)), "spw6": dict(p=dict(scale_pos_weight=6)),
     "es_hr": dict(es="hr"), "spw3_es": dict(p=dict(scale_pos_weight=3), es="hr"), "focal": dict(focal=True)}


def run(s, cfg):
    oof = np.zeros(len(y))
    for k, (a, b) in enumerate(folds_for(y, s)):
        r = np.random.default_rng(s + k); a = r.permutation(a); c = int(0.9 * len(a))
        p = dict(LGB_BASE); p.update(cfg.get("p", {}))
        if cfg.get("focal"): p["objective"] = focal_obj()
        if cfg.get("es") == "hr" or cfg.get("focal"): p["metric"] = "None"
        m = lgb.LGBMClassifier(n_estimators=5000, learning_rate=0.05, random_state=s + k, **p)
        m.fit(X.iloc[a[:c]], y[a[:c]], eval_set=[(X.iloc[a[c:]], y[a[c:]])],
              eval_metric=lgb_pauc if (cfg.get("es") == "hr" or cfg.get("focal")) else "auc",
              callbacks=[lgb.early_stopping(200, verbose=False)])
        oof[b] = m.predict(X.iloc[b], raw_score=True) if cfg.get("focal") else m.predict_proba(X.iloc[b])[:, 1]
    if cfg.get("focal"): oof = expit(oof)
    return oof


for v in (sys.argv[1:] or list(V)):
    hr = []
    def mk(s, v=v):
        o = run(s, V[v]); hr.append(pauc_hr(y, o)); return o
    agg, o = multi_seed(f"exp22 lgb fe_v2 {v} (binary-F2 oriented)", y, mk)
    print("   pAUC_hr", round(np.mean(hr), 5), "+-", round(np.std(hr, ddof=1), 5), flush=True)
    np.save(f"{OUT}/exp22_{v}.npy", np.array(o))

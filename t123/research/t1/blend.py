"""Nested blending of saved per-seed OOF matrices (shape (3, n) each, seeds = common.SEEDS).

usage: python blend.py name1=out/file1.npy name2=out/file2.npy ...
Compares: each model alone, equal rank-average, AUC-optimised rank weights (nested), LR stacking on logits (nested).
Level-2 models are fitted on 4/5 of the rows and scored on the 5th (folds seeded seed+1), so the blend score is honest;
thresholds are nested again inside report().
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from scipy.stats import rankdata
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression


def rank01(p): return rankdata(p) / len(p)
def logit(p): p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))


def to_score(P):  # P: (n, m) -> standardised logit-like columns (mixnb outputs are already log-odds)
    Z = np.column_stack([(logit(c) if (c.min() >= 0 and c.max() <= 1) else c) for c in P.T])
    return (Z - Z.mean(0)) / Z.std(0)


def fit_w(R, y):
    k = R.shape[1]; f = lambda z: -roc_auc_score(y, R @ (np.exp(z) / np.exp(z).sum()))
    r = min((minimize(f, z0, method="Nelder-Mead", options=dict(maxiter=300)) for z0 in [np.zeros(k)]), key=lambda r: r.fun)
    w = np.exp(r.x); return w / w.sum()


def nested_blends(P, y, seed):
    R = np.column_stack([rank01(c) for c in P.T]); Z = to_score(P)
    out = {"rank_avg": R.mean(1), "w_auc": np.zeros(len(y)), "lr_stack": np.zeros(len(y))}
    ws = []
    for a, b in folds_for(y, seed + 1):
        w = fit_w(R[a], y[a]); ws.append(w); out["w_auc"][b] = R[b] @ w
        lr = LogisticRegression(C=1.0).fit(Z[a], y[a]); out["lr_stack"][b] = lr.predict_proba(Z[b])[:, 1]
    return out, np.mean(ws, 0)


if __name__ == "__main__":
    lab, unl, te, ids = load(); y = lab[TARGET].astype(int).values
    items = [a.split("=", 1) for a in sys.argv[1:] if "=" in a]
    tag = "+".join(n for n, _ in items)
    M = {n: np.load(p) for n, p in items}
    res = {}
    for si, s in enumerate(SEEDS):
        P = np.column_stack([M[n][si] for n, _ in items])
        bl, w = nested_blends(P, y, s)
        for k, v in bl.items(): res.setdefault(k, []).append(report(f"blend[{tag}] {k} [seed {s}]", y, v, seed=s, save=False))
        print("   mean AUC-weights", dict(zip([n for n, _ in items], np.round(w, 3))))
    for k, rs in res.items():
        agg = dict(name=f"blend[{tag}] {k}")
        for m in ["auc", "f2m", "f2m_nested", "f2b", "f2b_nested"]:
            v = np.array([r[m] for r in rs]); agg[m] = round(v.mean(), 5); agg[m + "_sd"] = round(v.std(ddof=1), 5)
        print("AGG", json.dumps(agg))
        with open(os.path.join(OUT, "results_ms.jsonl"), "a") as f: f.write(json.dumps(agg) + "\n")

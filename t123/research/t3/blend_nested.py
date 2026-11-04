# Nested ensemble weight blending over model families (all with the same 3-seed classifier p).
# Outer = the v4 5 folds: weights fitted on 4 folds' OOF, applied to the 5th -> honest score. Test weights fitted on all OOF.
# Methods: LAD (LP, w >= 0, on hurdle predictions) | greedy forward selection with replacement on NMAE (Caruana),
#          on hurdle predictions ("greedy_pred") and on log-quantiles before the hurdle rule ("greedy_q").
# usage: python t123/research/t3/blend_nested.py [--write METHOD]
import sys, os, argparse, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load, nmae, fold_scores, save_oof, hurdle_median
from stack_lad import lad_weights

OD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "oof")
ALPHAS = np.array([0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
FAM = {   # family -> member models (geo-mean of quantiles inside a family = seed bag)
    "cb_d4": "cbq_d4,cbq_d4s1,cbq_d4s2", "cb_d6": "cbq_d6,cbq_d6s1,cbq_d6s2", "cb_d3": "cbq_d3",
    "cb_d8_lr03": "cbq_lr03,cbq3_s1,cbq3_s2", "cb_d8_lr06": "cbq_prev,cbq_s1,cbq_s2",
    "cb_d4_var": "cb_l2_10,cb_lr03", "xgb_d5": "xq_d5,xq_d5s1,xq_d5s2", "xgb_d7": "xq_extra",
    "xgb_hist": "xq_hist", "lgb_off": "lgbq_off", "xgb_linfull": "xq_linfull", "nn": "nnq",
}
ap = argparse.ArgumentParser(); ap.add_argument("--write", default=None); a = ap.parse_args()
D = load(); y, F = D["y"], D["folds"]; n = len(y)
P = np.mean([np.load(os.path.join(OD, f"{q}_p.npy")) for q in ("lgbh_log", "clf_s1", "clf_s2")], 0)
names = list(FAM); LQ = {}; PR = {}
for f, mem in FAM.items():
    LQ[f] = np.mean([np.log(np.clip(np.load(os.path.join(OD, f"{m}_Q.npy")), 1e-3, None)) for m in mem.split(",")], 0)
    PR[f] = hurdle_median(P, np.exp(LQ[f]), ALPHAS)
    print(f"{f:12s} {nmae(y, PR[f][:n]):.5f}")
M = np.column_stack([PR[f] for f in names])


def greedy(score_fn, idx, steps=40):
    cnt = np.zeros(len(names)); best_s, best_c = -1, None
    for _ in range(steps):
        cand = []
        for k in range(len(names)):
            c = cnt.copy(); c[k] += 1; cand.append((score_fn(c / c.sum(), idx), k))
        s, k = max(cand); cnt[k] += 1
        if s > best_s: best_s, best_c = s, cnt.copy()
    return best_c / best_c.sum()


def pred_w(w, idx):            # weighted average of hurdle predictions
    return M[idx] @ w


def predq_w(w, idx):           # weighted geo-mean of quantiles, then hurdle rule
    L = sum(wi * LQ[f][idx] for wi, f in zip(w, names) if wi > 0)
    return hurdle_median(P[idx], np.exp(L), ALPHAS)


METHODS = {
    "equal_final(cb_d4)": lambda tr: np.eye(len(names))[names.index("cb_d4")],
    "lad_pred": lambda tr: lad_weights(M[tr], y[tr]),
    "greedy_pred": lambda tr: greedy(lambda w, i: nmae(y[i], pred_w(w, i)), tr),
    "greedy_q": lambda tr: greedy(lambda w, i: nmae(y[i], predq_w(w, i)), tr, steps=25),
}
ref = PR["cb_d4"][:n]; RES = {}
for mn, fit in METHODS.items():
    o = np.zeros(n); W = []
    for tr, va in F:
        w = fit(tr); W.append(w); o[va] = predq_w(w, va) if mn == "greedy_q" else pred_w(w, va)
    W = np.array(W); d = np.array([nmae(y[v], o[v]) - nmae(y[v], ref[v]) for _, v in F]); m, s = fold_scores(y, o, F)
    RES[mn] = (o, W)
    print(f"RESULT {mn:20s} nested OOF {nmae(y, o):.5f} fold {m:.5f} +- {s:.5f} | paired vs cb_d4 bag {d.mean():+.5f} +- {d.std():.5f}")
    nz = W.mean(0) > 0.005
    print("   mean w:", {f: round(float(v), 3) for f, v in zip(np.array(names)[nz], W.mean(0)[nz])},
          "| max fold-to-fold sd of a weight %.3f" % W.std(0).max())
if a.write:
    w = METHODS[a.write](np.arange(n)); allidx = np.arange(len(P))
    full = predq_w(w, allidx) if a.write == "greedy_q" else pred_w(w, allidx)
    save_oof(f"blend_{a.write}", RES[a.write][0], full[n:])
    print("test weights:", {f: round(float(v), 3) for f, v in zip(names, w) if v > 0})

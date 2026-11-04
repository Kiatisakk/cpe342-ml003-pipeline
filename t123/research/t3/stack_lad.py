# Nested LAD (median-regression) stacking / blending over base OOF predictions.
# Outer: the v4 5 folds. For each outer fold the blend weights are fitted on the other 4 folds' OOF only,
# so the reported score is an honest nested estimate. Final test weights are fitted on all OOF.
# usage: python t123/research/t3/stack_lad.py base1 base2 ... [--v4]  (names in oof/, v4:<col> for v4 columns)
import sys, os, numpy as np, pandas as pd
from scipy.optimize import linprog
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load, nmae, fold_scores, load_oof

V4 = "t123/preds_final/v4/preds"


def get(name):
    if name.startswith("v4:"):
        c = name[3:]
        o = pd.read_csv(f"{V4}/t3_v4_oof.csv")[c].values
        t = pd.read_csv(f"{V4}/t3_v4_{c}.csv")["pred"].values
        return o, t
    return load_oof(name)


def lad_weights(M, y, nonneg=True, sum1=False):
    """min sum |y - M w|, w >= 0 (LP). Optionally sum(w)=1."""
    n, k = M.shape
    # vars: w (k), u+ (n), u- (n)
    c = np.r_[np.zeros(k), np.ones(n), np.ones(n)]
    from scipy.sparse import hstack, identity, csr_matrix
    A_eq = hstack([csr_matrix(M), identity(n), -identity(n)]).tocsr()
    b_eq = y
    if sum1:
        A_eq = __import__("scipy.sparse", fromlist=["vstack"]).vstack([A_eq, csr_matrix(np.r_[np.ones(k), np.zeros(2 * n)])])
        b_eq = np.r_[y, 1.0]
    bounds = [(0 if nonneg else None, None)] * k + [(0, None)] * (2 * n)
    r = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    return r.x[:k]


def nested(names, Ms, y, folds, **kw):
    M = np.column_stack(Ms); o = np.zeros(len(y)); W = []
    for tr, va in folds:
        w = lad_weights(M[tr], y[tr], **kw); W.append(w); o[va] = M[va] @ w
    return o, np.array(W)


if __name__ == "__main__":
    names = [x for x in sys.argv[1:] if not x.startswith("--")]
    D = load(); y, F = D["y"], D["folds"]
    O, T = zip(*[get(nm) for nm in names])
    for nm, o in zip(names, O): print(f"{nm:20s} {nmae(y, o):.5f}")
    o, W = nested(names, O, y, F)
    m, s = fold_scores(y, o, F)
    print("nested LAD blend OOF %.5f  fold %.5f +- %.5f" % (nmae(y, o), m, s))
    print("weights per fold\n", np.round(W, 3))

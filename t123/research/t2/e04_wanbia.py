"""E04: WANBIA-style weighted NB, honest (meta-learner fit on inner-OOF NB log-lik blocks of the outer train set).
variants: w  = one weight per feature (shared over classes)
          wk = one weight per feature x class
          lr = full multinomial LR on all F*K centred log-liks
usage: python t123/research/t2/e04_wanbia.py <variant> [l2]"""
import sys; sys.path.insert(0, 't123/research/t2')
from runner import *
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
V = sys.argv[1]; L2 = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-3
F = len(FEATS)


def fit_w(L, yy, per_class):
    n = len(yy); Y = np.eye(K)[yy]
    shp = (F, K) if per_class else (F, 1)
    def f(p):
        w = p[:np.prod(shp)].reshape(shp); c = p[np.prod(shp):]
        S = (L * w[None]).sum(1) + c
        S = S - S.max(1, keepdims=True); P = np.exp(S); Z = P.sum(1, keepdims=True); P /= Z
        loss = -np.mean(np.log(P[np.arange(n), yy] + 1e-12)) + L2 * np.sum((w - 1) ** 2)
        G = (P - Y) / n                                  # (n,K)
        gw = np.einsum('nfk,nk->fk', L, G)
        if not per_class: gw = gw.sum(1, keepdims=True)
        gw = gw + 2 * L2 * (w - 1)
        return loss, np.r_[gw.ravel(), G.sum(0)]
    p0 = np.r_[np.ones(np.prod(shp)), np.zeros(K)]
    r = minimize(f, p0, jac=True, method='L-BFGS-B', options=dict(maxiter=500))
    w = r.x[:np.prod(shp)].reshape(shp); c = r.x[np.prod(shp):]
    return lambda LL: softmax((LL * w[None]).sum(1) + c), w


def flat(L): return (L - L.mean(2, keepdims=True)).reshape(len(L), -1)


def fp(a, b):
    Lin, Lall, Lte = inner_blocks(a, seed=len(a) % 1000)
    if V in ('w', 'wk'):
        pred, w = fit_w(Lin.astype(np.float64), y[a], V == 'wk')
        return pred(Lall[b]), pred(Lte)
    m = LogisticRegression(C=L2, max_iter=2000).fit(flat(Lin), y[a])
    return m.predict_proba(flat(Lall[b])), m.predict_proba(flat(Lte))


if __name__ == '__main__':
    run(f'wanbia_{V}_{L2:g}', fp)

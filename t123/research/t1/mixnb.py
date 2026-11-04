"""Class-conditional mixture of naive-Bayes products (histogram densities), fitted by EM from scratch.

p(x | c) = sum_k pi_ck prod_f h_ckf(bin(x_f)); NaN features are marginalised (skipped).
Optional semi-supervised mode: unlabeled TRAIN rows enter EM with soft class responsibilities.
"""
import numpy as np
from scipy.special import logsumexp


class Binner:
    def __init__(self, n_bins=32): self.n_bins = n_bins
    def fit(self, X):  # X: (n, F) float, fitted on train rows only
        self.edges = []
        for j in range(X.shape[1]):
            v = X[:, j]; v = v[~np.isnan(v)]
            u = np.unique(v)
            if len(u) <= self.n_bins:  # discrete feature: one bin per value
                e = np.concatenate([[-np.inf], (u[1:] + u[:-1]) / 2, [np.inf]])
            else:
                e = np.unique(np.quantile(v, np.linspace(0, 1, self.n_bins + 1))); e[0], e[-1] = -np.inf, np.inf
            self.edges.append(e)
        self.sizes = [len(e) - 1 for e in self.edges]
        return self
    def transform(self, X):
        C = np.full(X.shape, -1, np.int32)
        for j, e in enumerate(self.edges):
            m = ~np.isnan(X[:, j]); C[m, j] = np.clip(np.searchsorted(e, X[m, j], side="right") - 1, 0, len(e) - 2)
        return C


class MixNB:
    def __init__(self, K=(4, 4), alpha=1.0, n_iter=60, seed=0, smooth=0):
        self.K, self.alpha, self.n_iter, self.seed, self.smooth = K, alpha, n_iter, seed, smooth

    def _loglik_comp(self, C, c):
        """(n, K_c) log pi + sum_f log h."""
        L = np.tile(np.log(self.pi[c])[None, :], (len(C), 1))
        for j in range(C.shape[1]):
            m = C[:, j] >= 0
            L[m] += self.logh[c][j][C[m, j]]          # logh[c][j]: (bins, K)
        return L

    def _mstep(self, C, W, c):
        """W: (n, K) responsibilities (already multiplied by class weight)."""
        K = W.shape[1]; self.pi[c] = (W.sum(0) + 1e-3) / (W.sum() + 1e-3 * K)
        hs = []
        for j in range(C.shape[1]):
            m = C[:, j] >= 0; B = self.sizes[j]
            H = np.stack([np.bincount(C[m, j], weights=W[m, k], minlength=B) for k in range(K)], 1) + self.alpha
            if self.smooth and B > 8:   # light smoothing along the bin axis
                kern = np.array([0.25, 0.5, 0.25]) if self.smooth == 1 else np.array([0.1, 0.2, 0.4, 0.2, 0.1])
                H = np.stack([np.convolve(np.pad(H[:, k], len(kern) // 2, mode="edge"), kern, "valid") for k in range(K)], 1)
            hs.append(np.log(H / H.sum(0, keepdims=True)))
        self.logh[c] = hs

    def fit(self, C, y, C_unl=None, prior_unl=None):
        rng = np.random.default_rng(self.seed); self.sizes = [int(max(C[:, j].max(), (C_unl[:, j].max() if C_unl is not None else 0))) + 1 for j in range(C.shape[1])]
        self.pi, self.logh = [None, None], [None, None]
        self.prior = np.array([1 - y.mean(), y.mean()])
        R = {}
        for c in (0, 1):
            Cc = C[y == c]; K = self.K[c]
            R[c] = rng.dirichlet(np.ones(K), len(Cc))
            self._mstep(Cc, R[c], c)
        for it in range(self.n_iter):
            if C_unl is not None:   # class posterior of unlabeled rows (semi-supervised EM)
                lu = np.stack([logsumexp(self._loglik_comp(C_unl, c), 1) + np.log(self.prior[c]) for c in (0, 1)], 1)
                qc = np.exp(lu - logsumexp(lu, 1, keepdims=True))
            for c in (0, 1):
                Cc = C[y == c]; L = self._loglik_comp(Cc, c); W = np.exp(L - logsumexp(L, 1, keepdims=True))
                if C_unl is not None:
                    Lu = self._loglik_comp(C_unl, c); Wu = np.exp(Lu - logsumexp(Lu, 1, keepdims=True)) * qc[:, [c]]
                    self._mstep(np.vstack([Cc, C_unl]), np.vstack([W, Wu]), c)
                else:
                    self._mstep(Cc, W, c)
        return self

    def decision(self, C):
        """log p(x|1) - log p(x|0) + log prior ratio."""
        l1 = logsumexp(self._loglik_comp(C, 1), 1); l0 = logsumexp(self._loglik_comp(C, 0), 1)
        return l1 - l0 + np.log(self.prior[1] / self.prior[0])

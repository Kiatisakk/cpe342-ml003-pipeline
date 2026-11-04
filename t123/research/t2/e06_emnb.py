"""E06: label-noise-aware Naive Bayes fitted by EM (train labels only, no player_id).
Model: latent true class z, observed label ~y with symmetric noise P(~y=k|z) = 1-eps if k==z else eps/3.
x_j independent given z (histogram densities, same binning/smoothing as the baseline per-feature config).
E-step: r_iz ~ pi_z prod_j p_j(x_ij|z) P(~y_i|z).   M-step: weighted histograms, pi, eps.
Prediction for a new row: P(z|x) ~ pi_z prod_j p_j(x_j|z); P(~y=k|x) = sum_z P(z|x)P(~y=k|z) (argmax identical).
usage: python t123/research/t2/e06_emnb.py [iters] [cfg: opt|base] [mode: em|idclean] [alpha]
  idclean = upper-bound diagnostic that trains on the ID-block labels (LEAK, diagnostic only)."""
import sys; sys.path.insert(0, 't123/research/t2')
from runner import *
from common import _disc_codes
IT = int(sys.argv[1]) if len(sys.argv) > 1 else 15
CF = sys.argv[2] if len(sys.argv) > 2 else 'opt'
MODE = sys.argv[3] if len(sys.argv) > 3 else 'em'
ALPHA = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
CFG = CFG_OPT if CF == 'opt' else CFG_BASE
ID = tr.player_id.str[1:].astype(int).values


def binned(a, cfg):
    """bin indices for all train rows and test rows per feature, edges from rows a. -1 = missing (marginalised)."""
    out = []
    for c in FEATS:
        if c in DISC:
            bt, be, n = _disc_codes(c, a)          # NaN is its own level for discrete (as baseline)
            sm = 0
        else:
            nb_, sm = cfg[c]
            x = tr[c].values.astype(float); xe = te[c].values.astype(float)
            e = np.unique(np.nanquantile(x[a], np.linspace(0, 1, nb_ + 1))); n = len(e) - 1
            bt = np.where(np.isnan(x), -1, np.clip(np.searchsorted(e, x, side='right') - 1, 0, n - 1))
            be = np.where(np.isnan(xe), -1, np.clip(np.searchsorted(e, np.nan_to_num(xe), side='right') - 1, 0, n - 1))
        out.append((bt, be, n, sm))
    return out


def densities(B, a, R, alpha):
    """log p_j(bin|z) tables from rows a with responsibilities R (len(a),K)"""
    tabs = []
    for bt, be, n, sm in B:
        ba = bt[a]; ok = ba >= 0; T = np.zeros((n, K))
        for k in range(K):
            cnt = np.bincount(ba[ok], weights=R[ok, k], minlength=n)
            if sm > 0: cnt = gaussian_filter1d(cnt, sm, mode='nearest')
            T[:, k] = np.log((cnt + alpha) / (cnt + alpha).sum())
        tabs.append(T)
    return tabs


def loglik(B, tabs, rows=None, test=False):
    S = 0
    for (bt, be, n, sm), T in zip(B, tabs):
        bb = be if test else bt[rows]
        L = T[np.maximum(bb, 0)]; L[bb < 0] = 0; S = S + L
    return S


def fit(a, B, labels=None):
    yy = y[a] if labels is None else labels[a]
    Y = np.eye(K)[yy]; R = Y.copy(); eps = 0.0; pi = Y.mean(0)
    its = IT if (MODE == 'em' and labels is None) else 0
    for it in range(its + 1):
        tabs = densities(B, a, R, ALPHA)
        if it == its: break
        S = loglik(B, tabs, a) + np.log(pi)
        e = max(eps, 1e-3) if it else 0.03       # start EM from a guess of 3% noise
        noise = np.where(Y > 0, np.log(1 - e), np.log(e / 3))
        R = softmax(S + noise)
        pi = R.mean(0)
        eps = 1 - (R * Y).sum() / len(a)
    return tabs, pi, eps


def make_fp(labels=None):
    def fp(a, b):
        B = binned(a, CFG)
        tabs, pi, eps = fit(a, B, labels)
        fp.eps.append(eps)
        M = np.full((K, K), eps / 3); np.fill_diagonal(M, 1 - eps)
        Pb = softmax(loglik(B, tabs, b) + np.log(pi)) @ M
        Pt = softmax(loglik(B, tabs, test=True) + np.log(pi)) @ M
        return Pb, Pt
    fp.eps = []
    return fp


if __name__ == '__main__':
    labels = None
    if MODE == 'idclean':
        labels = np.digitize(ID, [52000, 84500, 110500])
    fp = make_fp(labels)
    name = f'emnb_{MODE}_{CF}_it{IT}_a{ALPHA:g}'
    run(name, fp)
    print('estimated eps per fold:', np.round(fp.eps, 4))

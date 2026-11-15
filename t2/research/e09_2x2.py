"""E09: exploit the latent 2x2 structure (engagement axis {0,2} vs {1,3}, spending axis {0,1} vs {2,3}).
Factorised NB: each feature's class-conditional density is pooled over the axis it does NOT load on
(the axis is picked inside each training fold: the pooling with the higher held-out log-lik on an inner 80/20 split).
Fewer parameters -> lower variance, but biased if a feature carries small cross-axis effects."""
import sys; sys.path.insert(0, 't2/research')
from runner import *
from common import _disc_codes
import common as C
POOL = {'none': [0, 1, 2, 3], 'eng': [0, 1, 0, 1], 'spend': [0, 0, 1, 1]}   # class -> pooled group


def block_pooled(c, a, grp):
    """like loglik_block but densities estimated on pooled groups"""
    g = np.array(POOL[grp]); ysave = C.y.copy()
    C.y[:] = g[ysave]
    try:
        Lt, Le = loglik_block(c, a, *(CFG_OPT[c] if c in CONT else (0, 0)), 1.0)
    finally:
        C.y[:] = ysave
    return Lt[:, g], Le[:, g]   # group columns mapped back to the 4 classes (first G cols used)


def fp(a, b):
    rng = np.random.RandomState(0); m = rng.rand(len(a)) < 0.8; a1, a2 = a[m], a[~m]
    St = np.zeros((N, K)); Se = np.zeros((NT, K)); choice = []
    for c in FEATS:
        best = None
        for grp in POOL:
            Lt, _ = block_pooled(c, a1, grp)
            ll = np.mean(Lt[a2, y[a2]])
            if best is None or ll > best[0]: best = (ll, grp)
        choice.append(best[1])
        Lt, Le = block_pooled(c, a, best[1]); St += Lt; Se += Le
    fp.choices.append(choice)
    return softmax(St[b] + logprior(a)), softmax(Se + logprior(a))


fp.choices = []
if __name__ == '__main__':
    run('nb2x2', fp)
    from collections import Counter
    print({c: Counter(ch[j] for ch in fp.choices).most_common(1)[0] for j, c in enumerate(FEATS)})

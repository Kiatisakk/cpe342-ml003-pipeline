"""E00: reproduce baseline NB OOF macro-F1 (per-feature Optuna cfg, alpha=1) on seeds 42/7/123.
Also check equality with the original t123/t2_nb/nb.py implementation."""
import sys; sys.path.insert(0, 't123/research/t2')
from common import *

res = {}
for tag, cfg in (('base', CFG_BASE), ('opt', CFG_OPT)):
    sc = []
    for s in SEEDS:
        oof, _ = nb_blocks(cfg, 1.0, s, tag)
        S = oof.sum(1)
        for a, b in folds(s): S[b] += logprior(a)
        sc.append(f1m(y, S))
    res[tag] = report(f'NB {tag}', sc)

# cross-check with original code (seed 42, base cfg)
sys.path.insert(0, 't123/t2_nb'); import nb as ORIG
L = np.zeros((N, K))
for a, b in ORIG.folds: L[b] = ORIG.nb_scores(a, b, {'_default': dict(nbins=32, sm=0.7, alpha=1.0)})
print('original nb.py base seed42', round(ORIG.f1(L), 4))

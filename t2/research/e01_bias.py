"""E01: nested macro-F1 decision rule (additive class-bias on log-posterior) applied to baseline NB."""
import sys; sys.path.insert(0, 't2/research')
from common import *
plain, nest = [], []
for s in SEEDS:
    oof, _ = nb_blocks(CFG_OPT, 1.0, s, 'opt'); S = oof.sum(1)
    for a, b in folds(s): S[b] += logprior(a)
    plain.append(f1m(y, S)); f, bs = nested_bias_f1(S, s); nest.append(f)
    print(s, 'bias per fold', bs.round(2).tolist())
report('NB opt argmax', plain); report('NB opt + nested bias', nest)

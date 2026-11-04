"""E07: multi-resolution NB - average per-feature log-densities over several (nbins, sigma) histogram configs
(a cheap KDE-like smoother / variance reduction). Untuned (no F1 selection), so no nesting needed."""
import sys; sys.path.insert(0, 't123/research/t2')
from runner import *
CFGS = {'opt': CFG_OPT, 'base': CFG_BASE,
        'b16s1': {c: (16, 1.0) for c in CONT}, 'b24s1': {c: (24, 1.0) for c in CONT},
        'b48s15': {c: (48, 1.5) for c in CONT}, 'b64s2': {c: (64, 2.0) for c in CONT}}
for s in SEEDS:
    Ls, Ts = [], []
    for tag, cfg in CFGS.items():
        o, t_ = nb_blocks(cfg, 1.0, s, tag); Ls.append(o.sum(1)); Ts.append(t_.sum(1))
    S = np.mean(Ls, 0); T = np.mean(Ts, 0)
    for a, b in folds(s): S[b] += logprior(a)
    T = T + logprior(np.arange(N))
    np.savez_compressed(os.path.join(CACHE, f'P_nbbag_s{s}.npz'), oof=softmax(S).astype(np.float32), te=softmax(T))
    print(s, 'single cfgs:', [round(f1m(y, L), 4) for L in Ls], 'bag:', round(f1m(y, S), 4), 'bag+nested bias', round(nested_bias_f1(S, s)[0], 4), flush=True)
summary('nbbag')

"""E13: NESTED per-feature Optuna for the NB histogram config (honest estimate of the baseline's tuning).
For each outer fold of seed 42: Optuna (TPE, same search space as t2/code/pf.py) maximises inner 4-fold CV macro-F1
on the outer-train rows only; the chosen cfg is refit on the whole outer-train set and scored on the outer fold.
Also scores the untuned uniform cfg (32 bins, sigma 0.7) and the original (non-nested) Optuna cfg on the same folds.
usage: python t2/research/e13_nested_optuna.py <outer_fold 0-4> <n_trials>"""
import sys, json; sys.path.insert(0, 't2/research')
from common import *
import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING)
FO = int(sys.argv[1]); NT_ = int(sys.argv[2]) if len(sys.argv) > 2 else 150
NB = [8, 12, 16, 24, 32, 48, 64]; SM = [0, 0.5, 0.7, 1.0, 1.5, 2.0]
A, Bo = folds(42)[FO]
inner = list(StratifiedKFold(4, shuffle=True, random_state=FO).split(A, y[A]))
cache = {}
def LL(c, i, nb_, sm):
    k = (c, i, nb_, sm)
    if k not in cache:
        ia, ib = inner[i]; cache[k] = loglik_block(c, A[ia], nb_, sm, 1.0)[0][A[ib]].astype(np.float32)
    return cache[k]
def inner_f1(cfg):
    pred = np.zeros(len(A), int)
    for i, (ia, ib) in enumerate(inner):
        S = logprior(A[ia]) + sum(LL(c, i, 0, 0) if c in DISC else LL(c, i, *cfg[c]) for c in FEATS)
        pred[ib] = S.argmax(1)
    return f1_score(y[A], pred, average='macro')
def obj(tr_):
    return inner_f1({c: (tr_.suggest_categorical(f'nb_{c}', NB), tr_.suggest_categorical(f'sm_{c}', SM)) for c in CONT})
st = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=FO))
st.enqueue_trial({**{f'nb_{c}': 32 for c in CONT}, **{f'sm_{c}': 0.7 for c in CONT}})
st.optimize(obj, n_trials=NT_)
cfg = {c: (st.best_params[f'nb_{c}'], st.best_params[f'sm_{c}']) for c in CONT}
def outer_scores(cfg):
    S = logprior(A) + sum(loglik_block(c, A, *(cfg[c] if c in CONT else (0, 0)), 1.0)[0][Bo] for c in FEATS)
    return S
res = {'fold': FO, 'inner_base': st.trials[0].value, 'inner_best': st.best_value}
for name, cf in (('base', CFG_BASE), ('orig_optuna', CFG_OPT), ('nested_optuna', cfg)):
    S = outer_scores(cf); np.save(f'{CACHE}/e13_{name}_f{FO}.npy', S.astype(np.float32))
    res[name] = f1_score(y[Bo], S.argmax(1), average='macro')
res['cfg'] = cfg
print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items() if k != 'cfg'}), flush=True)
json.dump(res, open(f'{CACHE}/e13_f{FO}.json', 'w'))

"""Generic CV runner: fit_predict(a, b) -> (P_b, P_test) probabilities. 5-fold x seeds 42/7/123.
Saves per seed: OOF proba (N,K) and test proba averaged over that seed's 5 fold models (NT,K).
Set env T2_SEED=<seed> to run one seed only (lets seeds run as parallel processes)."""
import sys, time; sys.path.insert(0, 't123/research/t2')
from common import *
from sklearn.metrics import log_loss


def run(name, fit_predict, seeds=None):
    if seeds is None:
        seeds = (int(os.environ['T2_SEED']),) if 'T2_SEED' in os.environ else SEEDS
    for s in seeds:
        t0 = time.time(); oof = np.zeros((N, K)); tst = np.zeros((NT, K))
        for a, b in folds(s):
            Pb, Pt = fit_predict(a, b)
            oof[b] = Pb
            if Pt is not None: tst += Pt / 5
        np.savez_compressed(os.path.join(CACHE, f'P_{name}_s{s}.npz'), oof=oof.astype(np.float32), te=tst)
        print(f'{name} seed {s}: F1 {f1m(y, oof):.4f} LL {log_loss(y, np.clip(oof, 1e-9, 1)):.4f} ({time.time()-t0:.0f}s)', flush=True)
    if len(seeds) == len(SEEDS): return summary(name)


def load(name):
    zs = [np.load(os.path.join(CACHE, f'P_{name}_s{s}.npz')) for s in SEEDS]
    return np.stack([z['oof'] for z in zs]), np.mean([z['te'] for z in zs], 0)


def summary(name):
    oof, _ = load(name)
    report(f'{name} F1', [f1m(y, o) for o in oof])
    report(f'{name} logloss', [log_loss(y, np.clip(o, 1e-9, 1)) for o in oof])
    return oof

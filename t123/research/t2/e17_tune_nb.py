"""E17a: NESTED Optuna tuning of the final model's global hyperparameters (features only).
Model = E14 g2+EM (per-feature bins/sigma already chosen per outer fold, cache/nb_llsel_g2_em1_cfg_s<seed>.json).
Search space: alpha_cont (Laplace, continuous), alpha_disc (Laplace, discrete), sm_mult (x the selected sigma),
              em_iters, eps_mode (learned by EM | fixed 0.03 | none).
Objective = inner 4-fold CV LOG-LOSS on the outer-train rows only (smooth; F1 tuning overfits, see E13).
The chosen params are refit on the outer-train set and scored on the outer fold -> honest estimate.
usage: python t123/research/t2/e17_tune_nb.py <seed> [n_trials]"""
import sys, json; sys.argv_ = sys.argv[:]; sys.argv = sys.argv[:1] + ['15', 'opt', 'em', '1.0']
sys.path.insert(0, 't123/research/t2')
from runner import *
import e06_emnb as E
import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING)
SEED = int(sys.argv_[1]); NTR = int(sys.argv_[2]) if len(sys.argv_) > 2 else 30
CFGS = json.load(open(f'{CACHE}/nb_llsel_g2_em1_cfg_s{SEED}.json'))
DEF = dict(alpha_cont=1.0, alpha_disc=1.0, sm_mult=1.0, em_iters=15, eps_mode='em')


def model(A, cfg, p):
    """fit on rows A; returns predict(rows or 'test') -> P(observed label)"""
    cf = {c: (v[0], v[1] * p['sm_mult']) for c, v in cfg.items()}
    B = E.binned(A, cf); Y = np.eye(K)[y[A]]; R = Y.copy(); pi = Y.mean(0)
    eps = {'em': 0.03, 'fixed': 0.03, 'none': 0.0}[p['eps_mode']]
    def dens(R):
        tabs = []
        for (bt, be, n, sm), c in zip(B, FEATS):
            ba = bt[A]; ok = ba >= 0; T = np.zeros((n, K)); al = p['alpha_disc'] if c in DISC else p['alpha_cont']
            for k in range(K):
                cnt = np.bincount(ba[ok], weights=R[ok, k], minlength=n)
                if sm > 0: cnt = gaussian_filter1d(cnt, sm, mode='nearest')
                T[:, k] = np.log((cnt + al) / (cnt + al).sum())
            tabs.append(T)
        return tabs
    its = p['em_iters'] if p['eps_mode'] != 'none' else 0
    for it in range(its + 1):
        tabs = dens(R)
        if it == its: break
        S = E.loglik(B, tabs, A) + np.log(pi)
        R = softmax(S + np.where(Y > 0, np.log(1 - eps), np.log(eps / 3))); pi = R.mean(0)
        if p['eps_mode'] == 'em': eps = float(np.clip(1 - (R * Y).sum() / len(A), 1e-3, 0.2))
    M = np.full((K, K), eps / 3); np.fill_diagonal(M, 1 - eps)
    def pred(rows):
        S = E.loglik(B, tabs, test=True) if isinstance(rows, str) else E.loglik(B, tabs, rows)
        return softmax(S + np.log(pi)) @ M
    return pred


oof = np.zeros((N, K)); oof_def = np.zeros((N, K)); te_ = np.zeros((NT, K)); chosen = []
for fo, (A, Bo) in enumerate(folds(SEED)):
    cfg = CFGS[fo]
    inner = list(StratifiedKFold(4, shuffle=True, random_state=fo).split(A, y[A]))
    def obj(tr_):
        p = dict(alpha_cont=tr_.suggest_float('alpha_cont', 0.05, 10, log=True), alpha_disc=tr_.suggest_float('alpha_disc', 0.05, 30, log=True),
                 sm_mult=tr_.suggest_float('sm_mult', 0.5, 2.0), em_iters=tr_.suggest_int('em_iters', 3, 30),
                 eps_mode=tr_.suggest_categorical('eps_mode', ['em', 'fixed', 'none']))
        ll = 0
        for ia, ib in inner:
            Pb = model(A[ia], cfg, p)(A[ib]); ll += log_loss(y[A[ib]], np.clip(Pb, 1e-12, 1)) / 4
        return ll
    st = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=fo))
    st.enqueue_trial(DEF); st.optimize(obj, n_trials=NTR)
    p = st.best_params; chosen.append(p)
    f = model(A, cfg, p); oof[Bo] = f(Bo); te_ += f('test') / 5
    oof_def[Bo] = model(A, cfg, DEF)(Bo)
    print(f'seed {SEED} fold {fo}: inner LL default {st.trials[0].value:.5f} best {st.best_value:.5f} | outer F1 default {f1m(y[Bo], oof_def[Bo]):.4f} tuned {f1m(y[Bo], oof[Bo]):.4f} | {p}', flush=True)
np.savez_compressed(f'{CACHE}/P_nb_tuned_s{SEED}.npz', oof=oof.astype(np.float32), te=te_)
json.dump(chosen, open(f'{CACHE}/nb_tuned_params_s{SEED}.json', 'w'), indent=1)
print(f'seed {SEED}: F1 default {f1m(y, oof_def):.4f} LL {log_loss(y, oof_def):.4f} | tuned F1 {f1m(y, oof):.4f} LL {log_loss(y, oof):.4f}', flush=True)

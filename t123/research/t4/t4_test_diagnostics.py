# T4 follow-up research on saved probabilities (no GPU, no images). Run from repo root:
#   python t123/research/t4/t4_test_diagnostics.py [--reps 200]
# 1) Label-free estimates of test accuracy / macro-F1 (ATC, DoC, calibrated confidence, posterior sampling), each
#    validated first on OOF resamples where the truth is known.
# 2) Fold-ensemble analysis on test: agreement of the 5 fold models, arithmetic vs geometric mean.
# 3) Where the uncertainty is: top-2 class pairs of uncertain test rows vs OOF error pairs -> what v4 should target.
# 4) Expected-macro-F1-optimal decoding (per-class weights chosen to maximise posterior-expected macro-F1 on the
#    unlabelled set), with the same honest OOF simulation as t4_prior_shift.py.
# Test data use: predictions only (no labels exist). Writes results_diagnostics.json.
import argparse, json, itertools
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold

ap = argparse.ArgumentParser(); ap.add_argument("--reps", type=int, default=200); a = ap.parse_args()
D = "t123/preds_final/t4/preds"; OUTD = "t123/research/t4"; eps = 1e-12
tr = pd.read_csv("cpe342-kaggle-main/04_image_classification/train.csv")
C = sorted(tr["class"].unique()); K = len(C); y = tr["class"].map({c: i for i, c in enumerate(C)}).values
O = np.load(f"{D}/v3_5f_oof.npy").astype(float); P = np.load(f"{D}/task4_v3_5f_test_proba.npy").astype(float)
TF = np.stack([np.load(f"{D}/v3_5f_test_fold{k}.npy") for k in range(5)]).astype(float)
mf1 = lambda yt, pr: f1_score(yt, pr, average="macro", labels=range(K))
pi_tr = np.bincount(y, minlength=K) / len(y); R = {}

def nll(L, yy): L = L - L.max(1, keepdims=True); return -(L[np.arange(len(yy)), yy] - np.log(np.exp(L).sum(1))).mean()
def fit_T(p, yy): return minimize_scalar(lambda t: nll(np.log(p + eps) / t, yy), bounds=(0.05, 5), method="bounded").x
def temp(p, T): L = np.log(p + eps) / T; L -= L.max(1, keepdims=True); E = np.exp(L); return E / E.sum(1, keepdims=True)
def em_prior(p, pi, it=1000):
    q = pi.copy()
    for _ in range(it):
        w = p * (q / pi); w /= w.sum(1, keepdims=True); qn = w.mean(0)
        if np.abs(qn - q).max() < 1e-8: return qn
        q = qn
    return q
def correct(p, pi, q): w = p * (q / pi); return w / w.sum(1, keepdims=True)

# ------------------------------------------------ label-free performance estimators
def est_all(p_src, y_src, p_tgt, rng, T, nsamp=200):
    """p_src/y_src: labelled source (OOF) for fitting thresholds; p_tgt: unlabelled target. Returns dict of estimates."""
    acc_src = (p_src.argmax(1) == y_src).mean(); c_src, c_tgt = p_src.max(1), p_tgt.max(1)
    t = np.quantile(c_src, 1 - acc_src)                                        # ATC (Garg et al. 2022), max-confidence score
    out = {"ATC_acc": float((c_tgt > t).mean()), "DoC_acc": float(acc_src - (c_src.mean() - c_tgt.mean()))}
    q = temp(p_tgt, T); out["TS_conf_acc"] = float(q.max(1).mean())
    q = correct(q, pi_tr, em_prior(q, pi_tr))                                  # posterior after label-shift correction
    pr = p_tgt.argmax(1); cum = q.cumsum(1); f = []
    for _ in range(nsamp):                                                     # sample labels from the posterior
        ys = (rng.random(len(q))[:, None] > cum).sum(1).clip(0, K - 1); f.append(mf1(ys, pr))
    out["posterior_macroF1"] = float(np.mean(f)); out["posterior_acc"] = float((q[np.arange(len(q)), pr]).mean())
    return out

rng = np.random.default_rng(0)
halves = list(StratifiedKFold(2, shuffle=True, random_state=7).split(O, y))
bins = np.quantile(O.max(1), np.linspace(0, 1, 21)); bins[0], bins[-1] = 0, 1.01
wc = (np.histogram(P.max(1), bins)[0] / len(P) + 1e-6) / (np.histogram(O.max(1), bins)[0] / len(O))
roww = wc[np.clip(np.digitize(O.max(1), bins) - 1, 0, len(wc) - 1)]
q_target = np.array([0.1545, 0.2558, 0.146, 0.2727, 0.1711])                   # EM-BCTS test prior from t4_prior_shift.py
def draw(b, conf, n):
    cnt = rng.multinomial(n, q_target); idx = []
    for k in range(K):
        pool = b[y[b] == k]; w = roww[pool] if conf else np.ones(len(pool)); idx.append(rng.choice(pool, cnt[k], p=w / w.sum()))
    return np.concatenate(idx)

# 1a) validate the estimators on OOF resamples where the truth is known
val = {}
for scen, conf in [("label_shift", False), ("label_shift+conf", True)]:
    rows = []
    for rep in range(40):
        A, B = halves[rep % 2]; T = fit_T(O[A], y[A]); idx = draw(B, conf, len(P))
        e = est_all(O[A], y[A], O[idx], rng, T, nsamp=40)
        e["true_acc"] = float((O[idx].argmax(1) == y[idx]).mean()); e["true_macroF1"] = float(mf1(y[idx], O[idx].argmax(1))); rows.append(e)
    df = pd.DataFrame(rows); val[scen] = {c: [round(df[c].mean(), 4), round(df[c].std(), 4)] for c in df}
    val[scen]["bias_vs_truth"] = {"ATC_acc": round((df.ATC_acc - df.true_acc).mean(), 4), "DoC_acc": round((df.DoC_acc - df.true_acc).mean(), 4),
                                  "TS_conf_acc": round((df.TS_conf_acc - df.true_acc).mean(), 4), "posterior_acc": round((df.posterior_acc - df.true_acc).mean(), 4),
                                  "posterior_macroF1": round((df.posterior_macroF1 - df.true_macroF1).mean(), 4)}
    print(f"[validate {scen}] mean,sd:", val[scen])
R["estimator_validation_on_OOF"] = val
# 1b) apply to the real test (source = all OOF)
T_all = fit_T(O, y); R["T_full_oof"] = round(T_all, 4)
R["test_estimates"] = {k: round(v, 4) for k, v in est_all(O, y, P, rng, T_all).items()}
print("[test] label-free estimates:", R["test_estimates"])

# ------------------------------------------------ 2) fold ensemble on test
fp = TF.argmax(2); ens = P.argmax(1)
agree_all = (fp == fp[0]).all(0); n_agree = (fp == ens).sum(0)
G = np.exp(np.log(TF + eps).mean(0)); G /= G.sum(1, keepdims=True)
R["test_fold"] = {"all5_agree": round(agree_all.mean(), 4), "single_fold_vs_ens_agreement": [round((fp[k] == ens).mean(), 4) for k in range(5)],
                  "rows_by_votes_for_ens_label": {int(v): int((n_agree == v).sum()) for v in range(1, 6)},
                  "geo_vs_arith_label_changes": int((G.argmax(1) != ens).sum()),
                  "pairwise_fold_agreement_mean": round(np.mean([(fp[i] == fp[j]).mean() for i, j in itertools.combinations(range(5), 2)]), 4)}
print("[test folds]", R["test_fold"])

# ------------------------------------------------ 3) where is the uncertainty
def pairs(p, mask):
    top2 = np.argsort(-p[mask], 1)[:, :2]; s = pd.Series([" / ".join(sorted((C[i], C[j]))) for i, j in top2]).value_counts(normalize=True)
    return s.round(3).head(8).to_dict()
err = O.argmax(1) != y
oof_err_pairs = pd.Series([" / ".join(sorted((C[t], C[p]))) for t, p in zip(y[err], O.argmax(1)[err])]).value_counts(normalize=True).round(3).head(8).to_dict()
unc = P.max(1) < 0.6
R["uncertainty"] = {"oof_n_errors": int(err.sum()), "oof_error_pairs_share": oof_err_pairs,
                    "oof_share_maxprob_lt_0.6": round((O.max(1) < 0.6).mean(), 4), "test_share_maxprob_lt_0.6": round(unc.mean(), 4),
                    "test_uncertain_top2_pairs_share": pairs(P, unc),
                    "test_uncertain_by_pred_class": {C[k]: round(((P.argmax(1) == k) & unc).sum() / max(1, (P.argmax(1) == k).sum()), 4) for k in range(K)},
                    "oof_uncertain_by_true_class": {C[k]: round(((y == k) & (O.max(1) < 0.6)).sum() / (y == k).sum(), 4) for k in range(K)},
                    "test_fold_disagree_by_pred_class": {C[k]: round((~agree_all[ens == k]).mean(), 4) for k in range(K)}}
print("[uncertainty]", json.dumps(R["uncertainty"], indent=1))

# ------------------------------------------------ 4) expected-macro-F1 decoding, honest simulation
def exp_f1(q, pred):                                                          # ratio-of-expectations approximation
    f = []
    for k in range(K):
        m = pred == k; tp = q[m, k].sum(); fp_ = m.sum() - tp; fn = q[~m, k].sum(); f.append(2 * tp / max(eps, 2 * tp + fp_ + fn))
    return np.mean(f)
GRID = np.linspace(-1.5, 1.5, 31)
def fit_ef1(q, rounds=2):
    lq = np.log(q + eps); b = np.zeros(K); best = exp_f1(q, lq.argmax(1))
    for _ in range(rounds):
        for k in range(1, K):
            for g in GRID:
                bb = b.copy(); bb[k] = g; s = exp_f1(q, (lq + bb).argmax(1))
                if s > best + 1e-9: best, b = s, bb
    return b
sim = {}
for scen, conf in [("label_shift", False), ("label_shift+conf", True)]:
    d1, d2 = [], []
    for rep in range(a.reps):
        A, B = halves[rep % 2]; T = fit_T(O[A], y[A]); idx = draw(B, conf, len(P)); ys = y[idx]
        q = temp(O[idx], T); qe = correct(q, pi_tr, em_prior(q, pi_tr)); base = mf1(ys, O[idx].argmax(1))
        d1.append(mf1(ys, (np.log(qe + eps) + fit_ef1(qe)).argmax(1)) - base)   # EM posterior + expected-F1 weights
        d2.append(mf1(ys, (np.log(q + eps) + fit_ef1(q)).argmax(1)) - base)     # TS posterior only + expected-F1 weights
    d1, d2 = np.array(d1), np.array(d2)
    sim[scen] = {"EM+expF1": [round(d1.mean(), 5), round(d1.std(), 5), round((d1 > 0).mean(), 3)],
                 "TS+expF1": [round(d2.mean(), 5), round(d2.std(), 5), round((d2 > 0).mean(), 3)]}
    print(f"[expF1 decoding {scen}] delta vs raw argmax [mean, sd, P(>0)]:", sim[scen])
R["expected_F1_decoding_sim"] = sim; R["reps"] = a.reps
qT = temp(P, T_all); qTe = correct(qT, pi_tr, em_prior(qT, pi_tr))
R["test_expF1_labels_changed"] = {"TS+expF1": int(((np.log(qT + eps) + fit_ef1(qT)).argmax(1) != ens).sum()),
                                  "EM+expF1": int(((np.log(qTe + eps) + fit_ef1(qTe)).argmax(1) != ens).sum())}
print("[test] labels changed by expF1 decoding:", R["test_expF1_labels_changed"])
json.dump(R, open(f"{OUTD}/results_diagnostics.json", "w"), indent=1, default=float); print("wrote results_diagnostics.json")

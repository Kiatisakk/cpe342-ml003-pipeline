# T4 prior-shift (label-shift) correction on saved ConvNeXt-T DINOv3 5-fold probabilities.
# Run from repo root:  python t4/research/t4_prior_shift.py [--reps 200]
# Uses: OOF softmax + train labels (train only) and the *unlabelled* test softmax (only to estimate test priors).
# No test labels exist / are used. Writes t4/research/results_prior_shift.json and corrected test probabilities.
import argparse, json, os, time
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar, minimize
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

ap = argparse.ArgumentParser(); ap.add_argument("--reps", type=int, default=200); ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
D = "t4/preds/v3_5f/preds"; OUTD = "t4/research"
tr = pd.read_csv("cpe342-kaggle-main/04_image_classification/train.csv")
C = sorted(tr["class"].unique()); K = len(C)                       # datacard order: 7-11k fiveN freefried pube roblock
y = tr["class"].map({c: i for i, c in enumerate(C)}).values
O = np.load(f"{D}/v3_5f_oof.npy").astype(np.float64)
P = np.load(f"{D}/task4_v3_5f_test_proba.npy").astype(np.float64)
eps = 1e-12
mf1 = lambda yt, pr: f1_score(yt, pr, average="macro", labels=range(K))
R = {"classes": C}

# ---------------------------------------------------------------- calibration
# The model was trained with label smoothing 0.1 (+EMA, fold ensemble) -> probabilities are under-confident;
# EM needs calibrated posteriors, so fit temperature (TS) or bias-corrected TS (BCTS) on OOF by log-loss.
def nll(L, yy): L = L - L.max(1, keepdims=True); return -(L[np.arange(len(yy)), yy] - np.log(np.exp(L).sum(1))).mean()
def fit_cal(logp, yy, kind):
    if kind == "none": return np.array([1.0] + [0.0] * K)
    if kind == "ts":
        r = minimize_scalar(lambda t: nll(logp / t, yy), bounds=(0.05, 5), method="bounded"); return np.array([r.x] + [0.0] * K)
    r = minimize(lambda v: nll(logp / v[0] + v[1:], yy), np.array([0.5] + [0.0] * K), method="L-BFGS-B",
                 bounds=[(0.05, 5)] + [(-5, 5)] * K); return r.x
def apply_cal(logp, v):
    L = logp / v[0] + v[1:]; L = L - L.max(1, keepdims=True); E = np.exp(L); return E / E.sum(1, keepdims=True)

# ---------------------------------------------------------------- prior estimators
def em_prior(p, pi_tr, iters=1000, tol=1e-8):                          # Saerens et al. 2002
    q = pi_tr.copy()
    for _ in range(iters):
        a = p * (q / pi_tr); a /= a.sum(1, keepdims=True); qn = a.mean(0)
        if np.abs(qn - q).max() < tol: q = qn; break
        q = qn
    return q
def bbse_prior(p_src, y_src, p_tgt, soft=False):                     # Lipton et al. 2018 (hard) / soft confusion variant
    if soft: Cm = np.stack([p_src[y_src == j].mean(0) for j in range(K)], 1)       # Cm[i,j] = E[p_i | y=j]
    else:
        pr = p_src.argmax(1); Cm = np.zeros((K, K))
        for j in range(K): Cm[:, j] = np.bincount(pr[y_src == j], minlength=K) / max(1, (y_src == j).sum())
    mu = p_tgt.mean(0) if soft else np.bincount(p_tgt.argmax(1), minlength=K) / len(p_tgt)
    q = np.linalg.lstsq(Cm, mu, rcond=None)[0]; q = np.clip(q, 1e-6, None); return q / q.sum()
def correct(p, pi_tr, q):
    a = p * (q / pi_tr); return a / a.sum(1, keepdims=True)

# ---------------------------------------------------------------- per-class logit bias for macro-F1 (coordinate search)
GRID = np.linspace(-2, 2, 41)
def fit_bias(logp, yy, rounds=3):
    b = np.zeros(K); best = mf1(yy, (logp + b).argmax(1))
    for _ in range(rounds):
        for k in range(1, K):                                          # class 0 fixed (bias is shift-invariant)
            for g in GRID:
                bb = b.copy(); bb[k] = g; s = mf1(yy, (logp + bb).argmax(1))
                if s > best + 1e-9: best, b = s, bb
    return b

pi_tr = np.bincount(y, minlength=K) / len(y)
R["train_prior"] = pi_tr.round(4).tolist()
R["test_pred_share_argmax"] = (np.bincount(P.argmax(1), minlength=K) / len(P)).round(4).tolist()
R["test_mean_proba"] = P.mean(0).round(4).tolist()
R["oof_macroF1"] = round(mf1(y, O.argmax(1)), 5)
R["maxprob_quantiles_q01_q05_q10_q50"] = {"oof": np.quantile(O.max(1), [.01, .05, .1, .5]).round(3).tolist(),
                                          "test": np.quantile(P.max(1), [.01, .05, .1, .5]).round(3).tolist()}
print("train prior", R["train_prior"], "| test argmax share", R["test_pred_share_argmax"], "| OOF mF1", R["oof_macroF1"])

# ---------------------------------------------------------------- 1) calibration quality on OOF (cross-fitted)
lO, lP = np.log(O + eps), np.log(P + eps)
skf = list(StratifiedKFold(5, shuffle=True, random_state=1).split(O, y))
R["calibration_oof_nll"] = {}
for kind in ["none", "ts", "bcts"]:
    Q = np.zeros_like(O)
    for a, b in skf: Q[b] = apply_cal(lO[b], fit_cal(lO[a], y[a], kind))
    R["calibration_oof_nll"][kind] = round(nll(np.log(Q + eps), y), 5)
CAL = {k: fit_cal(lO, y, k) for k in ["none", "ts", "bcts"]}
R["cal_params_full_oof"] = {k: v.round(4).tolist() for k, v in CAL.items()}
print("OOF NLL (cross-fitted):", R["calibration_oof_nll"], "| T:", {k: round(v[0], 3) for k, v in CAL.items()})

# ---------------------------------------------------------------- 2) test prior estimates (unsupervised, test predictions only)
est = {}
for kind in ["none", "ts", "bcts"]:
    Ocal, Pcal = apply_cal(lO, CAL[kind]), apply_cal(lP, CAL[kind])
    pi_src = Ocal.mean(0) if kind == "none" else pi_tr                # for uncalibrated probs EM must use the model's own mean prior
    est[f"EM_{kind}"] = em_prior(Pcal, pi_src)
    est[f"BBSE_hard_{kind}"] = bbse_prior(Ocal, y, Pcal, soft=False)
    est[f"BBSE_soft_{kind}"] = bbse_prior(Ocal, y, Pcal, soft=True)
R["test_prior_estimates"] = {k: v.round(4).tolist() for k, v in est.items()}
for k, v in est.items(): print(f"  {k:16s}", v.round(4))

# ---------------------------------------------------------------- 3) honest simulation on OOF
# Split OOF in halves A/B (stratified). Fit calibration + bias on A only. From B draw n_test rows by class with the
# target prior (= our test prior estimate), run the WHOLE procedure blind (estimate prior on the sample, re-weight,
# argmax) and score macro-F1 before/after with the sample's true labels. Swap A/B. Repeat.
# Scenario "conf": additionally re-weights rows inside each class so the max-prob distribution matches the test
# (test is visibly less confident than OOF -> stress test for covariate shift on top of label shift).
n_te = len(P); target = est["EM_bcts"]
R["sim_target_prior"] = target.round(4).tolist()
bins = np.quantile(O.max(1), np.linspace(0, 1, 21)); bins[0], bins[-1] = 0, 1.01
w_conf = (np.histogram(P.max(1), bins)[0] / n_te + 1e-6) / (np.histogram(O.max(1), bins)[0] / len(O))
row_w_conf = w_conf[np.clip(np.digitize(O.max(1), bins) - 1, 0, len(w_conf) - 1)]
halves = list(StratifiedKFold(2, shuffle=True, random_state=7).split(O, y))
rng = np.random.default_rng(args.seed); t0 = time.time()
methods = ["raw", "cal_argmax", "bias", "EM", "EM+bias", "BBSE_hard", "BBSE_soft", "oracle"]
sim = {}
for cal_kind in ["ts", "bcts"]:
    fitted = []                                                        # per half: calibration + bias fitted on A only
    for a, b in halves:
        v = fit_cal(lO[a], y[a], cal_kind); Ca = apply_cal(lO[a], v); fitted.append((v, Ca, fit_bias(np.log(Ca + eps), y[a])))
    R[f"bias_fitted_on_halves_{cal_kind}"] = [f[2].round(2).tolist() for f in fitted]
    for scen in ["label_shift", "label_shift+conf", "no_shift"]:
        res = {m: [] for m in methods}; em_err = []
        for rep in range(args.reps):
            a, b = halves[rep % 2]; v, Ca, bias = fitted[rep % 2]
            tp = pi_tr if scen == "no_shift" else target
            cnt = rng.multinomial(n_te, tp); idx = []
            for k in range(K):
                pool = b[y[b] == k]; pw = row_w_conf[pool] if scen.endswith("conf") else np.ones(len(pool))
                idx.append(rng.choice(pool, cnt[k], replace=True, p=pw / pw.sum()))
            idx = np.concatenate(idx); ys = y[idx]; Ps = apply_cal(lO[idx], v); lPs = np.log(Ps + eps)
            q_em = em_prior(Ps, pi_tr); q_true = cnt / n_te; Pem = correct(Ps, pi_tr, q_em)
            preds = {"raw": O[idx].argmax(1), "cal_argmax": Ps.argmax(1), "bias": (lPs + bias).argmax(1),
                     "EM": Pem.argmax(1), "EM+bias": (np.log(Pem + eps) + bias).argmax(1),
                     "BBSE_hard": correct(Ps, pi_tr, bbse_prior(Ca, y[a], Ps, False)).argmax(1),
                     "BBSE_soft": correct(Ps, pi_tr, bbse_prior(Ca, y[a], Ps, True)).argmax(1),
                     "oracle": correct(Ps, pi_tr, q_true).argmax(1)}
            for m in methods: res[m].append(mf1(ys, preds[m]))
            em_err.append(float(np.abs(q_em - q_true).sum()))
        base = np.array(res["raw"]); out = {"raw_mean": float(round(base.mean(), 5)), "raw_sd": float(round(base.std(), 5)),
                                            "EM_prior_L1_error_mean": float(round(np.mean(em_err), 4))}
        for m in methods[1:]:
            d = np.array(res[m]) - base
            out[m] = {"mean": float(round(np.mean(res[m]), 5)), "delta_vs_raw": float(round(d.mean(), 5)), "sd": float(round(d.std(), 5)),
                      "P(delta>0)": float(round((d > 0).mean(), 3)), "P(delta<0)": float(round((d < 0).mean(), 3))}
        sim[f"{cal_kind}|{scen}"] = out
        print(f"\n[cal={cal_kind} | {scen}] reps={args.reps} raw {out['raw_mean']:.5f}+-{out['raw_sd']:.5f}  EM prior L1 err {out['EM_prior_L1_error_mean']}  ({time.time()-t0:.0f}s)")
        for m in methods[1:]: print(f"   {m:10s} {out[m]}")
R["simulation"] = sim; R["simulation_reps"] = args.reps

# ---------------------------------------------------------------- 4) nested per-class bias on OOF (no shift), 5-fold
R["nested_bias_oof"] = {"raw_argmax": float(round(mf1(y, O.argmax(1)), 5))}
for cal_kind in ["ts", "bcts"]:
    pr_b = np.zeros(len(y), int); Qc = np.zeros_like(O)
    for a, b in skf:
        v = fit_cal(lO[a], y[a], cal_kind); Qa, Qb = apply_cal(lO[a], v), apply_cal(lO[b], v); Qc[b] = Qb
        pr_b[b] = (np.log(Qb + eps) + fit_bias(np.log(Qa + eps), y[a])).argmax(1)
    R["nested_bias_oof"][f"{cal_kind}_argmax"] = float(round(mf1(y, Qc.argmax(1)), 5))
    R["nested_bias_oof"][f"{cal_kind}+bias"] = float(round(mf1(y, pr_b), 5))
print("\nnested 5-fold bias on OOF:", R["nested_bias_oof"])

# ---------------------------------------------------------------- 5) apply to test (full-OOF calibration) and save
raw_pred = P.argmax(1); R["test"] = {}
for cal_kind in ["ts", "bcts"]:
    Pc = apply_cal(lP, CAL[cal_kind]); q = em_prior(Pc, pi_tr); Pem = correct(Pc, pi_tr, q)
    np.save(f"{OUTD}/t4_test_proba_{cal_kind}_em.npy", Pem.astype(np.float32))
    R["test"][cal_kind] = {"EM_prior": q.round(4).tolist(),
                           "share_after_EM": (np.bincount(Pem.argmax(1), minlength=K) / n_te).round(4).tolist(),
                           "changed_by_calibration_only": int((Pc.argmax(1) != raw_pred).sum()),
                           "changed_by_cal+EM_vs_raw": int((Pem.argmax(1) != raw_pred).sum()),
                           "change_matrix_rows_raw_cols_new": confusion_matrix(raw_pred, Pem.argmax(1), labels=range(K)).tolist()}
    print(f"test [{cal_kind}]", R["test"][cal_kind])
json.dump(R, open(f"{OUTD}/results_prior_shift.json", "w"), indent=1)
print("wrote", f"{OUTD}/results_prior_shift.json")

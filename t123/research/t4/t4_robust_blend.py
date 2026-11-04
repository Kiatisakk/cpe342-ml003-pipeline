# T4 robust ensemble-weight blending pipeline (CPU, local). Run from repo root.
#
#   python t123/research/t4/t4_robust_blend.py --run v3_5f=t123/preds_final/t4/preds \
#          --run v4_eva02_s=t123/preds_final/t4_v4e/preds --run v4_cnxt_s_d3=t123/preds_final/t4_v4s/preds \
#          [--views plain,hflip] [--per-view] [--out t123/research/t4/blend]
#   python t123/research/t4/t4_robust_blend.py --selftest        # v3 + synthetic members, checks the selection logic
#
# Members: each --run TAG=DIR (v3/v4 output format: {TAG}_oof.npy, {TAG}_test_fold{0..4}.npy, v4 also *_{view}.npy).
#   --views picks TTA views of v4 runs (averaged per run); --per-view makes every (run, view) its own member.
#   The same TAG given twice = folds from several Kaggle sessions, merged. All runs must use the same 5 folds.
#
# Pipeline
#   1. per-member temperature scaling (fitted on OOF). Needed because members differ in sharpness
#      (v3 label smoothing 0.1, v4 0.05): without it, the sharper model dominates a probability average.
#   2. candidate blenders, all log-linear pooling  z = sum_m w_m * log q_m  (w on the simplex):
#        equal      w = 1/M
#        nll        w minimising OOF log-loss (smooth objective, less noisy than F1)
#        nll_bag    nll weights averaged over 20 bootstrap resamples of the OOF rows
#        caruana    greedy forward selection with replacement on log-loss (Caruana et al. 2004), bagged x10
#        shrink     0.5 * nll_bag + 0.5 * equal
#        f1grid     simplex grid (step 0.1) maximising OOF macro-F1 (only if M <= 4); the most overfit-prone
#   3. nested evaluation: outer 5-fold over OOF rows; calibration + weights fitted on 4/5, scored on 1/5 under
#        plain          the held-out rows as they are
#        shift          held-out rows resampled to the estimated test class mix
#        shift_conf     + re-weighted so the max-prob distribution matches the test (test is less confident)
#      Deltas vs `equal` are paired (same rows, same resamples).
#   4. robust selection: a method replaces `equal` only if its mean delta is >= 0 in EVERY scenario and its
#      plain-scenario delta is > 2 standard errors. Otherwise `equal` is kept (no weights to overfit).
#   5. refit the selected method on all OOF rows, apply it to the test, and write the outputs.
#   6. label-free diagnostics on test (ATC accuracy estimate per member and blend, member disagreement). These
#      are REPORTED ONLY and never used for selection (they are computed on test images).
# Test data use: predictions only. The class mix in step 3 is EM-estimated from test predictions (no labels exist).
# Outputs (in --out): t4_blend_test_proba.npy, t4_blend.csv (id,id_img,pred_num,pred_str), t4_blend_test_str.csv,
#   t4_blend_test_str_711K.csv, t4_blend_oof.npy (nested OOF of the selected method), blend_summary.json, BLEND_REPORT.md
import argparse, glob, itertools, json, os, time
import numpy as np, pandas as pd
from scipy.optimize import minimize, minimize_scalar
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

ap = argparse.ArgumentParser()
ap.add_argument("--run", action="append", default=[]); ap.add_argument("--views", default="")
ap.add_argument("--per-view", action="store_true"); ap.add_argument("--out", default="t123/research/t4/blend")
ap.add_argument("--resamples", type=int, default=20); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--selftest", action="store_true")
a = ap.parse_args()
eps = 1e-9; rng = np.random.default_rng(a.seed); t0 = time.time()
tr = pd.read_csv("cpe342-kaggle-main/04_image_classification/train.csv")
C = sorted(tr["class"].unique()); K = len(C); y = tr["class"].map({c: i for i, c in enumerate(C)}).values; N = len(y)
ref = pd.read_csv("t123/preds_final/t4/preds/task4_v3_5f.csv")             # id (ANS..), id_img in test.csv order
mf1 = lambda yt, z: f1_score(yt, z.argmax(1), average="macro", labels=range(K))

# ============================================================== 1. load members
def load_run(tag, d, views):
    has_views = bool(glob.glob(f"{d}/{tag}_oof_*.npy"))
    groups = [[f"_{v}"] for v in views] if (views and has_views and a.per_view) else \
             [[f"_{v}" for v in views]] if (views and has_views) else [[""]]
    out = {}
    for suf in groups:
        name = tag + ("" if suf == [""] else "|" + "+".join(s[1:] for s in suf))
        O = np.mean([np.load(f"{d}/{tag}_oof{s}.npy") for s in suf], 0)
        T = [np.mean([np.load(f"{d}/{tag}_test_fold{k}{s}.npy") for s in suf], 0) if os.path.exists(f"{d}/{tag}_test_fold{k}{suf[0]}.npy") else None
             for k in range(5)]
        out[name] = (O.astype(float), T)
    return out
members = {}
if a.selftest:                                                             # v3 + synthetic members (pipeline check only)
    d = "t123/preds_final/t4/preds"; O3 = np.load(f"{d}/v3_5f_oof.npy").astype(float)
    T3 = [np.load(f"{d}/v3_5f_test_fold{k}.npy").astype(float) for k in range(5)]
    def synth(p, sharp, noise, seed):
        r = np.random.default_rng(seed); L = np.log(p + eps) * sharp + r.normal(0, noise, p.shape)
        L -= L.max(1, keepdims=True); E = np.exp(L); return E / E.sum(1, keepdims=True)
    members["v3_5f"] = (O3, T3)
    members["synth_sharp_good"] = (synth(O3, 2.5, 0.8, 1), [synth(t, 2.5, 0.8, 10 + k) for k, t in enumerate(T3)])   # different sharpness, independent noise
    members["synth_bad"] = (synth(O3, 1.0, 3.0, 2), [synth(t, 1.0, 3.0, 20 + k) for k, t in enumerate(T3)])          # clearly worse
    a.out = a.out.rstrip("/") + "_selftest"
for spec in a.run:
    tag, d = spec.split("=", 1)
    for name, (O, T) in load_run(tag, d, [v for v in a.views.split(",") if v]).items():
        if name in members:                                                # merge partial Kaggle sessions
            O0, T0 = members[name]; O = np.where(np.isnan(O0), O, O0); T = [t0 if t0 is not None else t for t0, t in zip(T0, T)]
        members[name] = (O, T)
assert members, "give --run TAG=DIR (or --selftest)"
names = list(members); M = len(names)
for n in names:
    O, T = members[n]; miss = int(np.isnan(O[:, 0]).sum()); nt = sum(t is not None for t in T)
    assert O.shape == (N, K) and miss == 0 and nt == 5, f"{n}: shape {O.shape}, {miss} OOF rows missing, {nt}/5 test folds"
LO = np.stack([np.log(members[n][0] + eps) for n in names])                                   # M x N x K
LT = np.stack([np.log(np.mean(members[n][1], 0) + eps) for n in names])                       # M x Nt x K
print(f"{M} members:", {n: round(mf1(y, LO[i]), 5) for i, n in enumerate(names)}, flush=True)
if "v3_5f" in names: print("note: v3_5f OOF used best-epoch selection on its own fold (slightly optimistic); v4 OOF is honest")

# ============================================================== 2. calibration + blenders
def nll_of(z, yy): z = z - z.max(1, keepdims=True); return -(z[np.arange(len(yy)), yy] - np.log(np.exp(z).sum(1))).mean()
def fit_T(l, yy): return minimize_scalar(lambda t: nll_of(l / t, yy), bounds=(0.05, 5), method="bounded").x
def pool(Lc, w): return np.tensordot(w, Lc, 1)                                                  # Lc: M x n x K, calibrated logs
def logsm(z): z = z - z.max(-1, keepdims=True); return z - np.log(np.exp(z).sum(-1, keepdims=True))
def w_nll(Lc, yy):
    if M == 1: return np.ones(1)
    f = lambda th: nll_of(pool(Lc, np.exp(th) / np.exp(th).sum()), yy)
    th = minimize(f, np.zeros(M), method="L-BFGS-B").x; return np.exp(th) / np.exp(th).sum()
def w_bag(Lc, yy, fn, B):
    ws = []
    for _ in range(B):
        b = rng.integers(0, len(yy), len(yy)); ws.append(fn(Lc[:, b], yy[b]))
    return np.mean(ws, 0)
def w_caruana(Lc, yy, iters=25):
    cnt = np.zeros(M); cur = None
    for _ in range(iters):
        best = None
        for m in range(M):
            z = Lc[m] if cur is None else (cur * cnt.sum() + Lc[m]) / (cnt.sum() + 1); s = nll_of(z, yy)
            if best is None or s < best[0]: best = (s, m, z)
        cnt[best[1]] += 1; cur = best[2]
    return cnt / cnt.sum()
GRID = None
def w_f1grid(Lc, yy, step=0.1):
    global GRID
    if GRID is None:
        g = np.arange(0, 1 + 1e-9, step); GRID = np.array([w for w in itertools.product(g, repeat=M) if abs(sum(w) - 1) < 1e-6])
    return GRID[int(np.argmax([mf1(yy, pool(Lc, w)) for w in GRID]))]
METHODS = {"equal": lambda Lc, yy: np.full(M, 1 / M), "nll": w_nll,
           "nll_bag": lambda Lc, yy: w_bag(Lc, yy, w_nll, 20),
           "caruana": lambda Lc, yy: w_bag(Lc, yy, w_caruana, 10),
           "shrink": lambda Lc, yy: 0.5 * w_bag(Lc, yy, w_nll, 20) + 0.5 / M}
if M <= 4: METHODS["f1grid"] = w_f1grid
if M == 1: METHODS = {"equal": METHODS["equal"]}
def fit(method, idx):
    T = np.array([fit_T(LO[m, idx], y[idx]) for m in range(M)]); Lc = LO[:, idx] / T[:, None, None]
    return T, METHODS[method](Lc, y[idx])
def apply(T, w, L): return logsm(pool(L / T[:, None, None], w))

# ============================================================== 3. scenarios (class mix from test predictions, EM)
def em_prior(p, pi, it=1000):
    q = pi.copy()
    for _ in range(it):
        r = p * (q / pi); r /= r.sum(1, keepdims=True); qn = r.mean(0)
        if np.abs(qn - q).max() < 1e-8: return qn
        q = qn
    return q
pi_tr = np.bincount(y, minlength=K) / N
T_eq, _ = fit("equal", np.arange(N)); Zt_eq = apply(T_eq, np.full(M, 1 / M), LT); Zo_eq = apply(T_eq, np.full(M, 1 / M), LO)
q_test = em_prior(np.exp(Zt_eq), pi_tr)
bins = np.quantile(np.exp(Zo_eq).max(1), np.linspace(0, 1, 21)); bins[0], bins[-1] = 0, 1.01
dens = lambda v: np.histogram(v, bins)[0] / len(v)
wc = (dens(np.exp(Zt_eq).max(1)) + 1e-6) / (dens(np.exp(Zo_eq).max(1)) + 1e-6)
row_conf_w = wc[np.clip(np.digitize(np.exp(Zo_eq).max(1), bins) - 1, 0, len(wc) - 1)]
def resample(va, conf):
    cnt = rng.multinomial(len(va), q_test); idx = []
    for k in range(K):
        pool_ = va[y[va] == k]; w = row_conf_w[pool_] if conf else np.ones(len(pool_)); idx.append(rng.choice(pool_, cnt[k], p=w / w.sum()))
    return np.concatenate(idx)
print("EM test class mix (from test predictions):", dict(zip(C, q_test.round(3))), flush=True)

# ============================================================== 4. nested evaluation
outer = list(StratifiedKFold(5, shuffle=True, random_state=11).split(np.zeros(N), y))
scores = {m: {"plain": [], "shift": [], "shift_conf": []} for m in METHODS}
nll_plain = {m: [] for m in METHODS}; oof_nested = {m: np.zeros((N, K)) for m in METHODS}; wts = {m: [] for m in METHODS}
for fi, (trn, va) in enumerate(outer):
    fitted = {m: fit(m, trn) for m in METHODS}
    Z = {m: apply(*fitted[m], LO[:, va]) for m in METHODS}
    for m in METHODS:
        oof_nested[m][va] = Z[m]; wts[m].append(fitted[m][1]); scores[m]["plain"].append(mf1(y[va], Z[m])); nll_plain[m].append(nll_of(Z[m], y[va]))
    pos = {v: i for i, v in enumerate(va)}
    for scen, conf in [("shift", False), ("shift_conf", True)]:
        for _ in range(a.resamples):
            ridx = np.array([pos[i] for i in resample(va, conf)])
            for m in METHODS: scores[m][scen].append(mf1(y[va][ridx], Z[m][ridx]))
    print(f"outer fold {fi}: " + " ".join(f"{m} {scores[m]['plain'][-1]:.5f}" for m in METHODS) + f"  ({time.time()-t0:.0f}s)", flush=True)

summary = {"members": names, "single_member_oof_macroF1": {n: round(mf1(y, LO[i]), 5) for i, n in enumerate(names)},
           "test_class_mix_EM": q_test.round(4).tolist(), "methods": {}}
for m in METHODS:
    row = {"nested_oof_macroF1": round(mf1(y, oof_nested[m]), 5), "nested_oof_nll": round(float(np.mean(nll_plain[m])), 5),
           "weights_mean_over_folds": np.mean(wts[m], 0).round(3).tolist(), "weights_sd_over_folds": np.std(wts[m], 0).round(3).tolist()}
    for scen in ["plain", "shift", "shift_conf"]:
        d = np.array(scores[m][scen]) - np.array(scores["equal"][scen])
        row[scen] = {"macroF1": round(float(np.mean(scores[m][scen])), 5), "delta_vs_equal": round(float(d.mean()), 5),
                     "se": round(float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0, 5)}
    summary["methods"][m] = row

# ============================================================== 5. robust selection + refit + outputs
def ok(m):
    r = summary["methods"][m]
    return all(r[s]["delta_vs_equal"] >= 0 for s in ["plain", "shift", "shift_conf"]) and r["plain"]["delta_vs_equal"] > 2 * r["plain"]["se"]
cands = [m for m in METHODS if m != "equal" and ok(m)]
chosen = max(cands, key=lambda m: min(summary["methods"][m][s]["delta_vs_equal"] for s in ["plain", "shift", "shift_conf"])) if cands else "equal"
T_f, w_f = fit(chosen, np.arange(N)); Zt = apply(T_f, w_f, LT); PT = np.exp(Zt)
summary.update({"selected": chosen, "selection_rule": "delta>=0 in all scenarios and plain delta > 2 se, else equal",
                "final_temperatures": dict(zip(names, np.round(T_f, 4).tolist())), "final_weights": dict(zip(names, np.round(w_f, 4).tolist()))})

# ============================================================== 6. label-free test diagnostics (report only)
def atc(z_src, y_src, z_tgt):
    acc = (z_src.argmax(1) == y_src).mean(); t = np.quantile(z_src.max(1), 1 - acc); return float((z_tgt.max(1) > t).mean())
Lc_full = LO / T_f[:, None, None]; LTc = LT / T_f[:, None, None]
summary["test_diagnostics_report_only"] = {
    "ATC_acc_member": {n: round(atc(Lc_full[i], y, LTc[i]), 4) for i, n in enumerate(names)},
    "ATC_acc_blend": round(atc(oof_nested[chosen], y, Zt), 4),
    "member_pairwise_disagreement_test": {f"{names[i]} vs {names[j]}": round(float((LT[i].argmax(1) != LT[j].argmax(1)).mean()), 4)
                                          for i, j in itertools.combinations(range(M), 2)},
    "fold_disagreement_test_member": {n: round(float(np.mean([(members[n][1][i].argmax(1) != members[n][1][j].argmax(1)).mean()
                                          for i, j in itertools.combinations(range(5), 2)])), 4) for n in names},
    "test_share": dict(zip(C, (np.bincount(PT.argmax(1), minlength=K) / len(PT)).round(4).tolist())),
    "labels_changed_vs_first_member": int((PT.argmax(1) != LT[0].argmax(1)).sum())}

os.makedirs(a.out, exist_ok=True)
np.save(f"{a.out}/t4_blend_test_proba.npy", PT.astype(np.float32)); np.save(f"{a.out}/t4_blend_oof.npy", np.exp(oof_nested[chosen]).astype(np.float32))
lab = np.array(C)[PT.argmax(1)]
pd.DataFrame({"id": ref["id"], "id_img": ref["id_img"], "pred_num": PT.argmax(1), "pred_str": lab}).to_csv(f"{a.out}/t4_blend.csv", index=False, lineterminator="\n")
pd.DataFrame({"id": ref["id"], "pred": lab}).to_csv(f"{a.out}/t4_blend_test_str.csv", index=False, lineterminator="\n")
pd.DataFrame({"id": ref["id"], "pred": np.where(lab == "7-11k", "7-11K", lab)}).to_csv(f"{a.out}/t4_blend_test_str_711K.csv", index=False, lineterminator="\n")
json.dump(summary, open(f"{a.out}/blend_summary.json", "w"), indent=1)

md = [f"# T4 blend report{' (SELFTEST: synthetic members)' if a.selftest else ''}", "",
      "Members (single OOF macro-F1): " + ", ".join(f"`{n}` {v}" for n, v in summary["single_member_oof_macroF1"].items()), "",
      "Nested 5-fold OOF. Δ is paired vs `equal`, ± 1 se. `shift` = resampled to the EM test class mix; `shift_conf` = also matched to the test confidence.", "",
      "| method | nested OOF mF1 | NLL | Δ plain | Δ shift | Δ shift_conf | weights (mean ± sd over folds) |", "|---|---|---|---|---|---|---|"]
for m, r in summary["methods"].items():
    md.append(f"| {'**'+m+'**' if m == chosen else m} | {r['nested_oof_macroF1']:.5f} | {r['nested_oof_nll']:.4f} | "
              + " | ".join(f"{r[s]['delta_vs_equal']:+.5f} ± {r[s]['se']:.5f}" for s in ["plain", "shift", "shift_conf"])
              + " | " + ", ".join(f"{w:.2f}±{s:.2f}" for w, s in zip(r["weights_mean_over_folds"], r["weights_sd_over_folds"])) + " |")
md += ["", f"**Selected: `{chosen}`** (rule: {summary['selection_rule']}).",
       f"Final weights {summary['final_weights']}, temperatures {summary['final_temperatures']}.", "",
       "Label-free test diagnostics (report only, not used for selection): " + json.dumps(summary["test_diagnostics_report_only"])]
open(f"{a.out}/BLEND_REPORT.md", "w").write("\n".join(md) + "\n")
print("\n".join(md)); print(f"\nwrote {a.out}/ ({time.time()-t0:.0f}s) - nothing submitted")

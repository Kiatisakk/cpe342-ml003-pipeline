"""FINAL v2 T1 recipe (run from repo root:  python t1/research/final_v2.py).

= final.py (fe_v2 features, LightGBM lr 0.02 baseline-tuned params, 3 bags, 5-fold seeds 42/7/2024)
+ MCAR masking augmentation (exp16): in each fold, NC copies of the fold-TRAIN rows get extra NaNs (each signal value
  hidden with prob RATE x its observed missing rate) and fe_v2 is recomputed on them. Valid because missingness is
  MCAR (cheater rate | NaN = base rate for every column). Early-stopping holdout, held-out fold and test untouched.
Outputs (overwrite final.py's): best_oof.npy (seed-42 OOF probs), best_oof_allseeds.npy, best_test_proba.npy,
t1_best_test.csv (macro-F2 cut), t1_best_test_binaryF2.csv (binary-F2 cut), final_results.json.
Test cut = OOF-optimal positive rate of the 3-seed averaged OOF (make_test_files.py, exp17).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
import lightgbm as lgb

N_BAG = int(os.environ.get("N_BAG", 3)); NC = int(os.environ.get("NC", 4)); RATE = float(os.environ.get("RATE", 1.0))
LR = float(os.environ.get("LR", 0.02)); PREFIX = os.environ.get("PREFIX", "")


def main():
    T = Timer()
    lab, unl, te, ids = load(); y = lab[TARGET].astype(int).values
    ref = pd.concat([lab, unl])
    X, Xt = fe_v2(lab, ref), fe_v2(te, ref)
    miss_rate = lab[SIGNAL].isna().mean().values

    def masked_copy(df, rng):
        d = df.copy(); M = rng.random((len(d), len(SIGNAL))) < miss_rate[None, :] * RATE
        v = d[SIGNAL].values.copy(); v[M] = np.nan; d[SIGNAL] = v
        return fe_v2(d, ref)

    per = {}
    for s in SEEDS:
        oof, tp = np.zeros(len(y)), np.zeros(len(Xt))
        for i in range(N_BAG):
            for k, (a, b) in enumerate(folds_for(y, s)):
                rng = np.random.default_rng(s + k + 100 * i); a = rng.permutation(a); cut = int(0.9 * len(a)); tr, va = a[:cut], a[cut:]
                Xa = pd.concat([X.iloc[tr]] + [masked_copy(lab.iloc[tr], rng) for _ in range(NC)], ignore_index=True)
                m = lgb.LGBMClassifier(n_estimators=20000, learning_rate=LR, random_state=s + k + 1000 * i, **LGB_BASE)
                m.fit(Xa, np.tile(y[tr], NC + 1), eval_set=[(X.iloc[va], y[va])], eval_metric="auc",
                      callbacks=[lgb.early_stopping(200, verbose=False)])
                oof[b] += m.predict_proba(X.iloc[b])[:, 1] / N_BAG
                tp += m.predict_proba(Xt)[:, 1] / (5 * N_BAG)
        per[s] = (oof, tp); print(f"seed {s} done {T()}", flush=True)
    rs = [report(f"final_v2 (NC{NC} rate{RATE} lr{LR} bag{N_BAG}) [seed {s}]", y, per[s][0], seed=s, save=False) for s in SEEDS]
    summary = {k: [round(float(np.mean([r[k] for r in rs])), 5), round(float(np.std([r[k] for r in rs], ddof=1)), 5)]
               for k in ["auc", "f2m", "f2m_nested", "f2b", "f2b_nested"]}
    print("AGG", json.dumps(summary), flush=True)
    oofs = np.array([per[s][0] for s in SEEDS]); test = np.mean([per[s][1] for s in SEEDS], 0)
    P = lambda f: os.path.join(ROOT, PREFIX + f)
    np.save(P("best_oof.npy"), oofs[0]); np.save(P("best_oof_allseeds.npy"), oofs); np.save(P("best_test_proba.npy"), test)
    out = {"recipe": dict(NC=NC, RATE=RATE, LR=LR, N_BAG=N_BAG), "summary_mean_sd": summary}
    for kind, fn in [("macro", "t1_best_test.csv"), ("binary", "t1_best_test_binaryF2.csv")]:
        thr = [best_thr(y, oofs[i], kind) for i in range(len(SEEDS))]
        t_prob = float(np.mean([t[0] for t in thr])); rate = float(np.mean([t[2] for t in thr]))
        pred = (test >= np.quantile(test, 1 - rate)).astype(int)
        pd.DataFrame({"id": ids, "pred": pred, "score": test}).to_csv(P(fn), index=False)
        out[kind] = dict(threshold_prob=round(t_prob, 5), oof_pos_rate=round(rate, 4), test_pos_rate=round(float(pred.mean()), 4),
                         test_pos_rate_if_prob_cut=round(float((test >= t_prob).mean()), 4), oof_score_seed42=round(float(thr[0][1]), 5))
        print(kind, out[kind], flush=True)
    out["time"] = T()
    json.dump(out, open(P("final_results.json"), "w"), indent=1)
    from make_test_files import build
    build(PREFIX)   # final cuts from the 3-seed averaged OOF (matches the averaged test model)
    print("done", T(), "- nothing submitted")


if __name__ == "__main__":
    main()

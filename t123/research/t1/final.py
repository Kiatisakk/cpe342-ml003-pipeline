"""FINAL T1 recipe (run from repo root:  python t123/research/t1/final.py).

features : fe_v2 = baseline FE (task1_gbm_pipeline) + exact zone posteriors for missing age / level
           (label-free; prior fitted on train rows = labeled + unlabeled train; test only transformed)
models   : LightGBM lr 0.02 (baseline-tuned params), bagged over 3 inner early-stopping splits / model seeds
           (an XGB lr0.05 half was tested in v1 of this script: blend AUC 0.89386 < LGB alone 0.89412 -> dropped)
blend    : plain average of the 3 bags (no fitted weights -> nothing to nest)
CV       : 5-fold StratifiedKFold, fold-split seeds 42 / 7 / 2024 (42 = baseline scheme); thresholds nested
outputs  : best_oof.npy (seed-42 OOF probabilities, labeled-train file order), best_oof_allseeds.npy,
           t1_best_test.csv (macro-F2 threshold), t1_best_test_binaryF2.csv (binary-F2 threshold), final_results.json
Test cut: the OOF-optimal positive rate. Test rows are predicted by the average of all 5 folds x 3 seeds models.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

N_BAG = 3


def run_seed(X, y, Xt, s):
    lo, lt = np.zeros(len(y)), np.zeros(len(Xt))
    for i in range(N_BAG):
        o, t = cv_lgb(X, y, seed=s, lr=0.02, Xt=Xt, inner_seed_offset=100 * i, params=dict(seed=1000 * i + s))
        lo += o / N_BAG; lt += t / N_BAG
    return dict(blend=(lo, lt))


if __name__ == "__main__":
    T = Timer()
    lab, unl, te, ids = load(); y = lab[TARGET].astype(int).values
    ref = pd.concat([lab, unl])
    X, Xt = fe_v2(lab, ref), fe_v2(te, ref)
    per = {}
    for s in SEEDS:
        per[s] = run_seed(X, y, Xt, s); print(f"seed {s} done {T()}", flush=True)
    summary = {}
    for comp in ["blend"]:
        rs = [report(f"final {comp} [seed {s}]", y, per[s][comp][0], seed=s, save=False) for s in SEEDS]
        summary[comp] = {k: [round(float(np.mean([r[k] for r in rs])), 5), round(float(np.std([r[k] for r in rs], ddof=1)), 5)]
                         for k in ["auc", "f2m", "f2m_nested", "f2b", "f2b_nested"]}
        print("AGG", comp, json.dumps(summary[comp]), flush=True)
    oofs = np.array([per[s]["blend"][0] for s in SEEDS]); test = np.mean([per[s]["blend"][1] for s in SEEDS], 0)
    np.save(os.path.join(ROOT, "best_oof.npy"), oofs[0]); np.save(os.path.join(ROOT, "best_oof_allseeds.npy"), oofs)
    np.save(os.path.join(ROOT, "best_test_proba.npy"), test)
    out = {"summary_mean_sd": summary}
    for kind, fn in [("macro", "t1_best_test.csv"), ("binary", "t1_best_test_binaryF2.csv")]:
        thr = [best_thr(y, oofs[i], kind) for i in range(len(SEEDS))]
        t_prob = float(np.mean([t[0] for t in thr])); rate = float(np.mean([t[2] for t in thr]))
        # test cut = OOF-optimal positive rate (test is i.i.d. with train; averaging 15 models smooths the
        # probabilities, so a raw probability cut would shift the predicted rate). Probability cut kept for reference.
        pred = (test >= np.quantile(test, 1 - rate)).astype(int)
        pd.DataFrame({"id": ids, "pred": pred, "score": test}).to_csv(os.path.join(ROOT, fn), index=False)
        out[kind] = dict(threshold_prob=round(t_prob, 5), oof_pos_rate=round(rate, 4), test_pos_rate=round(float(pred.mean()), 4),
                         test_pos_rate_if_prob_cut=round(float((test >= t_prob).mean()), 4),
                         oof_score_seed42=round(float(thr[0][1]), 5))
        print(kind, out[kind], flush=True)
    out["time"] = T()
    json.dump(out, open(os.path.join(ROOT, "final_results.json"), "w"), indent=1)
    print("done", T(), "- nothing submitted")

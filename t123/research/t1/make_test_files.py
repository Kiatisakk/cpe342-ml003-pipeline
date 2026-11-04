"""Rebuild the test deliverables from saved predictions (no retraining).

The test score is the average of the 5-fold x 3-seed x 3-bag models, so the matching OOF is the average of the
3 split-seed OOFs (each row's 3 predictions come from models that never saw it). Cuts = OOF-optimal positive rate
of that averaged OOF (exp17: nested F2m 0.78279 +- 0.00013 vs 0.78201 for single-seed OOF).
usage (repo root): python t123/research/t1/make_test_files.py [PREFIX]
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *


def build(prefix=""):
    P = lambda f: os.path.join(ROOT, prefix + f)
    lab, unl, te, ids = load(); y = lab[TARGET].astype(int).values
    O = np.load(P("best_oof_allseeds.npy")); test = np.load(P("best_test_proba.npy")); avg = O.mean(0)
    np.save(P("best_oof_avg3.npy"), avg)
    res = json.load(open(P("final_results.json"))) if os.path.exists(P("final_results.json")) else {}
    res["avg3_oof"] = dict(auc=round(roc_auc_score(y, avg), 5))
    for kind, fn in [("macro", "t1_best_test.csv"), ("binary", "t1_best_test_binaryF2.csv")]:
        t, s, rate = best_thr(y, avg, kind)
        nested = [round(nested_thr_score(y, avg, folds_for(y, sd), kind), 5) for sd in SEEDS]
        pred = (test >= np.quantile(test, 1 - rate)).astype(int)
        pd.DataFrame({"id": ids, "pred": pred, "score": test}).to_csv(P(fn), index=False)
        res["avg3_oof"][kind] = dict(oof_score=round(s, 5), nested_per_seed=nested, nested_mean=round(float(np.mean(nested)), 5),
                                     nested_sd=round(float(np.std(nested, ddof=1)), 5), pos_rate=round(rate, 4), thr_prob=round(float(t), 4),
                                     test_pos_rate=round(float(pred.mean()), 4))
        print(kind, res["avg3_oof"][kind], flush=True)
    json.dump(res, open(P("final_results.json"), "w"), indent=1)


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "")

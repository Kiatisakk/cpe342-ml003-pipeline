# Blend T4 runs (v3_5f + v4 runs downloaded from Kaggle) on OOF, honestly. Run from repo root, e.g.
#   python t4/research/t4_blend_runs.py --run v3_5f=t4/preds/v3_5f/preds \
#          --run v4_cnxt_s_d3=t4/preds/v3_5f_v4s/preds --run v4_eva02_s=t4/preds/v3_5f_v4e/preds
# Each --run is TAG=DIR where DIR holds {TAG}_oof.npy and {TAG}_test_fold{0..4}.npy (v3/v4 output format).
# v4 runs also have per-view files ({TAG}_oof_{view}.npy, {TAG}_test_fold{k}_{view}.npy); with --views the TTA views are
# chosen per run. Folds split over several Kaggle sessions: pass the same TAG several times (NaN rows are merged).
# Blend weights: simplex grid on log-probabilities, scored with NESTED 5-fold CV over OOF rows (weights chosen on 4/5,
# scored on 1/5) so the reported number is not optimistic; final weights are refit on all OOF rows.
# Writes OUT/t4_blend_test_proba.npy, OUT/t4_blend.csv (id,id_img,pred_num,pred_str), OUT/t4_blend_test_str.csv (id,pred).
import argparse, glob, itertools, json, os
import numpy as np, pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

ap = argparse.ArgumentParser()
ap.add_argument("--run", action="append", required=True); ap.add_argument("--views", default="")
ap.add_argument("--out", default="t4/research/blend"); ap.add_argument("--step", type=float, default=0.1)
a = ap.parse_args()
tr = pd.read_csv("cpe342-kaggle-main/04_image_classification/train.csv")
C = sorted(tr["class"].unique()); y = tr["class"].map({c: i for i, c in enumerate(C)}).values; K = len(C)
ref = pd.read_csv("t4/preds/v3_5f/preds/task4_v3_5f.csv")             # id (ANS..) + id_img in test.csv order
mf1 = lambda yt, p: f1_score(yt, p.argmax(1), average="macro")
views = [v for v in a.views.split(",") if v]

runs = {}
for spec in a.run:
    tag, d = spec.split("=", 1)
    suf = [f"_{v}" for v in views] if views and glob.glob(f"{d}/{tag}_oof_*.npy") else [""]
    O = np.mean([np.load(f"{d}/{tag}_oof{s}.npy") for s in suf], 0)
    T = [np.mean([np.load(f"{d}/{tag}_test_fold{k}{s}.npy") for s in suf], 0) if os.path.exists(f"{d}/{tag}_test_fold{k}{suf[0]}.npy") else None
         for k in range(5)]
    if tag in runs:                                                          # merge partial sessions
        O0, T0 = runs[tag]; O = np.where(np.isnan(O0), O, O0); T = [t0 if t0 is not None else t for t0, t in zip(T0, T)]
    runs[tag] = (O, T)
tags = list(runs)
for t in tags:
    O, T = runs[t]; miss = np.isnan(O[:, 0]).sum(); nt = sum(x is not None for x in T)
    assert miss == 0 and nt == 5, f"{t}: {miss} OOF rows missing, {nt}/5 test folds"
    print(f"{t:16s} OOF macroF1 {mf1(y, O):.5f}")
L = np.stack([np.log(runs[t][0] + 1e-9) for t in tags]); LT = np.stack([np.log(np.mean(runs[t][1], 0) + 1e-9) for t in tags])

g = np.arange(0, 1 + 1e-9, a.step)
W = np.array([w for w in itertools.product(g, repeat=len(tags)) if abs(sum(w) - 1) < 1e-6])
def best_w(idx): return W[int(np.argmax([mf1(y[idx], np.tensordot(w, L[:, idx], 1)) for w in W]))]
pred = np.zeros((len(y), K))
for tr_i, va_i in StratifiedKFold(5, shuffle=True, random_state=3).split(y, y):
    pred[va_i] = np.tensordot(best_w(tr_i), L[:, va_i], 1)
w = best_w(np.arange(len(y)))
eq = np.tensordot(np.full(len(tags), 1 / len(tags)), L, 1)
print(f"equal-weight blend OOF {mf1(y, eq):.5f} | nested-CV weighted blend {mf1(y, pred):.5f} | full-OOF weights {dict(zip(tags, w.round(2)))}")

os.makedirs(a.out, exist_ok=True)
PT = np.exp(np.tensordot(w, LT, 1)); PT /= PT.sum(1, keepdims=True); np.save(f"{a.out}/t4_blend_test_proba.npy", PT.astype(np.float32))
lab = np.array(C)[PT.argmax(1)]
pd.DataFrame({"id": ref["id"], "id_img": ref["id_img"], "pred_num": PT.argmax(1), "pred_str": lab}).to_csv(f"{a.out}/t4_blend.csv", index=False, lineterminator="\n")
pd.DataFrame({"id": ref["id"], "pred": lab}).to_csv(f"{a.out}/t4_blend_test_str.csv", index=False, lineterminator="\n")
json.dump({"runs": tags, "weights": w.tolist(), "oof_single": {t: mf1(y, runs[t][0]) for t in tags},
           "oof_equal": mf1(y, eq), "oof_nested_weighted": mf1(y, pred),
           "test_share": dict(zip(C, (np.bincount(PT.argmax(1), minlength=K) / len(PT)).round(4).tolist()))},
          open(f"{a.out}/blend_summary.json", "w"), indent=1)
print("test share", dict(zip(C, (np.bincount(PT.argmax(1), minlength=K) / len(PT)).round(3))), "| wrote", a.out)
